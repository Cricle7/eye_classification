import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class IrisNet(nn.Module):
    def __init__(self, num_classes=3):
        super(IrisNet, self).__init__()

        # 量化模块
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # 特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 40x40 -> 20x20

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 20x20 -> 10x10

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 10x10 -> 5x5
        )

        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 量化输入
        x = self.quant(x)

        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平
        x = self.classifier(x)

        # 反量化输出
        x = self.dequant(x)
        return x
