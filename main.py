import torch
from torch import nn, optim
from datasets import IrisDataset
from models import IrisNet  # 假设你把模型文件命名为 model.py
from train import IrisTrainer  # 假设你有一个训练类 IrisTrainer
from torch.optim import Adam


def main():
    # 初始化设备、模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IrisNet(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # 初始化数据集
    dataset = IrisDataset('data/iris')

    # 初始化训练器
    trainer = IrisTrainer(dataset, model, criterion, optimizer, device)

    # 开始训练并执行 K 折交叉验证
    avg_accuracy = trainer.train_kfold(k=5)
    print(f"Average accuracy over 5 folds: {avg_accuracy:.4f}")


if __name__ == "__main__":
    main()
