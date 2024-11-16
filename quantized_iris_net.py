# models/quantized_iris_net.py
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class QuantizedIrisNet(nn.Module):
    def __init__(self, num_classes=3):
        super(QuantizedIrisNet, self).__init__()
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

            nn.MaxPool2d(2),  # 28 -> 14
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # 配置量化
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self, inplace=True)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x
