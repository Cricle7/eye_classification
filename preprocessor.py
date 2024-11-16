from torchvision import transforms
from PIL import Image

# 定义 Preprocessor 类
class Preprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((40, 40)),
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])

    def preprocess(self, iris_img):
        if iris_img.size == 0:
            return None

        # 转换为 PIL 图像
        iris_pil = Image.fromarray(iris_img).convert('L')

        # 应用预处理
        iris_tensor = self.transform(iris_pil)
        iris_tensor = iris_tensor.unsqueeze(0)  # 添加批次维度

        return iris_tensor