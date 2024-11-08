import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class IrisNet(nn.Module):
    def __init__(self, num_classes=3):
        super(IrisNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 输入通道数为1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class IrisRecognizer:
    def __init__(self, model_path='iris_model.pth', label_mapping=None):
        """
        初始化 IrisRecognizer 类。

        参数:
        - model_path: 训练好的模型文件路径。
        - label_mapping: 标签映射的字典，键为标签索引，值为对应的人员名称。
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # 加载模型
        self.num_classes = len(label_mapping)
        self.model = IrisNet(num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 标签映射
        self.label_mapping = label_mapping

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])

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

    def process_frame(self, image):
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

                # 进行预测
                left_prediction = self.predict(left_input)
                right_prediction = self.predict(right_input)

                # 获取预测结果的人员名称
                left_person = self.label_mapping[left_prediction]
                right_person = self.label_mapping[right_prediction]

                # 在图像上绘制边界框和人员名称
                cv2.rectangle(image, (left_bbox[0], left_bbox[1]), (left_bbox[2], left_bbox[3]), (0, 255, 0), 2)
                cv2.putText(image, left_person, (left_bbox[0], left_bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.rectangle(image, (right_bbox[0], right_bbox[1]), (right_bbox[2], right_bbox[3]), (0, 255, 0), 2)
                cv2.putText(image, right_person, (right_bbox[0], right_bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                recognition_results.append((left_person, right_person))

        return image, recognition_results

    def preprocess_iris(self, iris_img):
        """
        对虹膜图像进行预处理。
        """
        if iris_img.size == 0:
            return None

        # 转换为 PIL 图像
        iris_pil = Image.fromarray(cv2.cvtColor(iris_img, cv2.COLOR_BGR2RGB)).convert('L')

        # 应用与训练时相同的预处理
        iris_tensor = self.transform(iris_pil)
        iris_tensor = iris_tensor.unsqueeze(0).to(self.device)  # 添加批次维度

        return iris_tensor

    def predict(self, input_tensor):
        """
        使用模型进行预测。
        """
        if input_tensor is None:
            return None

        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.item()

    def run(self):
        """
        启动实时识别流程。
        """
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    print("无法读取摄像头数据")
                    break

                processed_frame, recognition_results = self.process_frame(frame)

                # 显示结果
                cv2.imshow('Iris Recognition', processed_frame)

                # 按下 'Esc' 键退出
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # 定义标签映射（根据您的实际人员名称）
    label_mapping = {
        0: 'person1',
        1: 'person2',
        2: 'person3'
    }

    recognizer = IrisRecognizer(
        model_path='iris_model.pth',
        label_mapping=label_mapping
    )
    recognizer.run()
