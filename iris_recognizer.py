# real_time_recognition.py

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import models  # 确保 models.py 在同一目录下
from tqdm import tqdm
from serial_communicator import SerialCommunicator
from preprocessor import Preprocessor


# 定义 IrisRecognizer 类
class IrisRecognizer:
    def __init__(self, model_path='final_iris_model.pth', label_mapping=None):
        """
        初始化 IrisRecognizer 类。

        参数:
        - model_path: 训练好的模型文件路径。
        - label_mapping: 标签映射的字典，键为标签索引，值为对应的人员名称。
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # 加载模型
        self.num_classes = 3
        self.model = models.IrisNet(num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 标签映射
        self.label_mapping = label_mapping

        # 初始化预处理器
        self.preprocessor = Preprocessor()

        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 定义虹膜的关键点索引
        self.IRIS_LEFT_IDX = [468, 469, 470, 471, 472]
        self.IRIS_RIGHT_IDX = [473, 474, 475, 476, 477]

        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("无法打开摄像头")

        # 初始化串口通信
        self.serial_comm = SerialCommunicator(
            serial_port='COM32',  # 根据您的实际串口设备
            baud_rate=115200,
            timeout=1
        )

    def get_iris_region(self, landmarks, indices, image_shape):
        """
        提取虹膜的关键点坐标。
        """
        iris = []
        for idx in indices:
            x = int(landmarks[idx].x * image_shape[1])
            y = int(landmarks[idx].y * image_shape[0])
            iris.append((x, y))
        return iris

    def get_iris_bbox(self, iris, image_shape):
        """
        计算虹膜的边界框。
        """
        xs = [point[0] for point in iris]
        ys = [point[1] for point in iris]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        # 扩展边界框以包含整个虹膜区域
        padding_x = int((x_max - x_min) * 0.5)
        padding_y = int((y_max - y_min) * 0.5)
        return (max(x_min - padding_x, 0),
                max(y_min - padding_y, 0),
                min(x_max + padding_x, image_shape[1]),
                min(y_max + padding_y, image_shape[0]))

    def preprocess_iris(self, iris_img):
        """
        对虹膜图像进行预处理。
        """
        iris_tensor = self.preprocessor.preprocess(iris_img)
        if iris_tensor is not None:
            iris_tensor = iris_tensor.to(self.device)
        return iris_tensor

    def predict(self, input_tensor):
        """
        使用模型进行预测，返回预测的标签和对应的置信度。
        """
        if input_tensor is None:
            return 'unknown', 0.0

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
            predicted_label = self.label_mapping.get(predicted.item(), 'unknown')
            return predicted_label, max_prob.item()

    def process_frame(self, image, threshold=0.5):
        """
        处理单帧图像，检测虹膜并进行识别。
        """
        # 将图像从 BGR 转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理图像并获取面部关键点
        results = self.face_mesh.process(image_rgb)

        recognition_results = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 提取左眼和右眼的虹膜关键点
                left_iris = self.get_iris_region(face_landmarks.landmark, self.IRIS_LEFT_IDX, image.shape)
                right_iris = self.get_iris_region(face_landmarks.landmark, self.IRIS_RIGHT_IDX, image.shape)

                # 计算左眼和右眼的边界框
                left_bbox = self.get_iris_bbox(left_iris, image.shape)
                right_bbox = self.get_iris_bbox(right_iris, image.shape)

                # 提取虹膜图像
                left_iris_img = image[left_bbox[1]:left_bbox[3], left_bbox[0]:left_bbox[2]]
                right_iris_img = image[right_bbox[1]:right_bbox[3], right_bbox[0]:right_bbox[2]]

                # 对虹膜图像进行预处理
                left_input = self.preprocess_iris(left_iris_img)
                right_input = self.preprocess_iris(right_iris_img)

                # 对左右眼分别进行预测，获取标签和置信度
                left_person, left_confidence = self.predict(left_input)
                right_person, right_confidence = self.predict(right_input)

                # 选取置信度较高的预测结果
                if left_confidence >= right_confidence:
                    final_person = left_person
                    final_confidence = left_confidence
                else:
                    final_person = right_person
                    final_confidence = right_confidence

                # 判断置信度是否高于阈值
                if final_confidence < threshold:
                    final_person = 'unknown'

                # 在左右眼的边界框上绘制相同的标签
                cv2.rectangle(image, (left_bbox[0], left_bbox[1]), (left_bbox[2], left_bbox[3]), (0, 255, 0), 2)
                cv2.putText(image, f"{final_person} ({final_confidence:.2f})", (left_bbox[0], left_bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.rectangle(image, (right_bbox[0], right_bbox[1]), (right_bbox[2], right_bbox[3]), (0, 255, 0), 2)
                cv2.putText(image, f"{final_person} ({final_confidence:.2f})", (right_bbox[0], right_bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                recognition_results.append(final_person)

                # 在每次识别后发送数据
                self.serial_comm.send_data(final_person)

        return image, recognition_results

    def run(self, threshold=0.5):
        """
        启动实时识别流程。
        """
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    print("无法读取摄像头数据")
                    break

                processed_frame, recognition_results = self.process_frame(frame, threshold)

                # 显示结果
                cv2.imshow('Iris Recognition', processed_frame)

                # 按下 'Esc' 键退出
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            # 关闭串口连接
            self.serial_comm.close()

if __name__ == "__main__":
    # 定义标签映射（根据您的实际人员名称）
    label_mapping = {
        0: 'person1',
        1: 'person2',
        2: 'person3'
    }

    recognizer = IrisRecognizer(
        model_path='iris_model_epoch_800.pth',
        label_mapping=label_mapping
    )
    recognizer.run()
