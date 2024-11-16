import cv2
import mediapipe as mp
import numpy as np
import os

class IrisDataCollector:
    def __init__(self, save_dir='iris_dataset', person_id='person_1', camera_id=0):
        """
        初始化 IrisDataCollector 类。
        
        参数:
        - save_dir: 保存虹膜图像的根目录。
        - person_id: 当前采集的人员 ID，用于分类存储图像。
        - camera_id: 摄像头的设备编号，默认值为 0。
        """
        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 定义眼睛的关键点索引
        self.LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
        self.RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]
        self.IRIS_LEFT_IDX = [468, 469, 470, 471, 472]
        self.IRIS_RIGHT_IDX = [473, 474, 475, 476, 477]
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise IOError("无法打开摄像头")
        
        # 设置保存路径
        self.save_dir = save_dir
        self.person_id = person_id
        self.left_eye_dir = os.path.join(save_dir, person_id, 'left_eye')
        self.right_eye_dir = os.path.join(save_dir, person_id, 'right_eye')
        os.makedirs(self.left_eye_dir, exist_ok=True)
        os.makedirs(self.right_eye_dir, exist_ok=True)
        
        # 图像计数器
        self.left_eye_count = 0
        self.right_eye_count = 0
    
    def get_eye_region(self, landmarks, indices, image_shape):
        """
        提取眼睛的关键点坐标。
        """
        eye = []
        for idx in indices:
            x = int(landmarks[idx].x * image_shape[1])
            y = int(landmarks[idx].y * image_shape[0])
            eye.append((x, y))
        return eye
    
    def get_eye_bbox(self, eye, image_shape):
        """
        计算眼睛的边界框。
        """
        xs = [point[0] for point in eye]
        ys = [point[1] for point in eye]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        # 扩展边界框以包含整个虹膜
        padding_x = int((x_max - x_min) * 0.3)
        padding_y = int((y_max - y_min) * 0.3)
        return (max(x_min - padding_x, 0),
                max(y_min - padding_y, 0),
                min(x_max + padding_x, image_shape[1]),
                min(y_max + padding_y, image_shape[0]))
    
    def process_frame(self, image):
        """
        处理单帧图像，定位虹膜并保存虹膜区域。
        """
        # 将图像从 BGR 转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 处理图像并获取面部关键点
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 提取左眼和右眼的关键点
                left_iris = self.get_eye_region(face_landmarks.landmark, self.IRIS_LEFT_IDX, image.shape)
                right_iris = self.get_eye_region(face_landmarks.landmark, self.IRIS_RIGHT_IDX, image.shape)
                
                # 计算左眼和右眼的边界框
                left_bbox = self.get_eye_bbox(left_iris, image.shape)
                right_bbox = self.get_eye_bbox(right_iris, image.shape)
                
                # 提取虹膜图像
                left_iris_img = image[left_bbox[1]:left_bbox[3], left_bbox[0]:left_bbox[2]]
                right_iris_img = image[right_bbox[1]:right_bbox[3], right_bbox[0]:right_bbox[2]]
                
                # 保存虹膜图像
                left_eye_filename = os.path.join(self.left_eye_dir, f'left_iris_{self.left_eye_count}.png')
                right_eye_filename = os.path.join(self.right_eye_dir, f'right_iris_{self.right_eye_count}.png')
                cv2.imwrite(left_eye_filename, left_iris_img)
                cv2.imwrite(right_eye_filename, right_iris_img)
                self.left_eye_count += 1
                self.right_eye_count += 1
                
                # 在原图上绘制边界框（可选）
                cv2.rectangle(image, (left_bbox[0], left_bbox[1]), (left_bbox[2], left_bbox[3]), (0, 255, 0), 2)
                cv2.rectangle(image, (right_bbox[0], right_bbox[1]), (right_bbox[2], right_bbox[3]), (0, 255, 0), 2)
                
        return image
    
    def run(self):
        """
        启动虹膜数据采集流程。
        """
        print("按下 's' 键开始采集，按下 'q' 键退出。")
        collecting = False
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    print("无法读取摄像头数据")
                    break
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):
                    collecting = True
                    print("开始采集虹膜图像...")
                elif key == ord('q'):
                    break
                
                if collecting:
                    processed_frame = self.process_frame(frame)
                else:
                    processed_frame = frame.copy()
                
                cv2.imshow('Iris Data Collection', processed_frame)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print(f"采集完成，共保存了 {self.left_eye_count} 张左眼虹膜图像和 {self.right_eye_count} 张右眼虹膜图像。")

if __name__ == "__main__":
    collector = IrisDataCollector(person_id='person_2_1')
    collector.run()
