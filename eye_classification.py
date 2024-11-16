from iris_recognizer import IrisRecognizer
from utils import label_mapping

def eye_classification():
    recognizer = IrisRecognizer(
        model_path='models/quantized_iris_model.pth',  # 确保使用量化后的模型
        label_mapping=label_mapping,
        serial_port='COM22',
        baud_rate=115200,
        timeout=1
    )
    recognizer.run(threshold=0.5)  # 设置置信度阈值为0.5

if __name__ == "__eye_classification__":
    eye_classification()
