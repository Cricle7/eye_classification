# models.py
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub


class IrisNet(nn.Module):
    def __init__(self, num_classes=3):
        super(IrisNet, self).__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.conv1x1 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.conv1x1(x)
        x = self.dequant(x)
        return x.view(x.size(0), -1)  # Flatten the output
