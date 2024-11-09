# models.py
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class IrisNet(nn.Module):
    def __init__(self, num_classes=3):
        super(IrisNet, self).__init__()

        # 添加 QuantStub 和 DeQuantStub 用于量化
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28

            # 添加额外的池化层以进一步下采样
            nn.MaxPool2d(2),  # 28 -> 14
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 14 * 14, 512),  # 从25088降到512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # 输出层
        )

    def forward(self, x):
        # 量化输入
        x = self.quant(x)

        x = self.features(x)
        # 移除 avgpool
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        # 反量化输出
        x = self.dequant(x)
        return x
