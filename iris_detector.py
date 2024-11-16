import mediapipe as mp
import cv2

class IrisDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.IRIS_LEFT_IDX = [468, 469, 470, 471, 472]
        self.IRIS_RIGHT_IDX = [473, 474, 475, 476, 477]

    def get_iris_region(self, landmarks, indices, image_shape):
        iris = []
        for idx in indices:
            x = int(landmarks[idx].x * image_shape[1])
            y = int(landmarks[idx].y * image_shape[0])
            iris.append((x, y))
        return iris

    def get_iris_bbox(self, iris, image_shape):
        xs = [point[0] for point in iris]
        ys = [point[1] for point in iris]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        padding_x = int((x_max - x_min) * 0.5)
        padding_y = int((y_max - y_min) * 0.5)
        return (
            max(x_min - padding_x, 0),
            max(y_min - padding_y, 0),
            min(x_max + padding_x, image_shape[1]),
            min(y_max + padding_y, image_shape[0])
        )

    def detect_iris(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        irises = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_iris = self.get_iris_region(face_landmarks.landmark, self.IRIS_LEFT_IDX, image.shape)
                right_iris = self.get_iris_region(face_landmarks.landmark, self.IRIS_RIGHT_IDX, image.shape)

                left_bbox = self.get_iris_bbox(left_iris, image.shape)
                right_bbox = self.get_iris_bbox(right_iris, image.shape)

                irises.append({
                    'left': {
                        'bbox': left_bbox,
                        'image': image[left_bbox[1]:left_bbox[3], left_bbox[0]:left_bbox[2]]
                    },
                    'right': {
                        'bbox': right_bbox,
                        'image': image[right_bbox[1]:right_bbox[3], right_bbox[0]:right_bbox[2]]
                    }
                })

        return irises
