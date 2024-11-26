import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datasets import IrisDataset
from models import IrisNet


class IrisTrainer:
    def __init__(self, dataset, model, criterion, optimizer, device):
        self.dataset = dataset
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels)
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds.double() / total_preds
        return epoch_loss, epoch_accuracy

    def evaluate(self, val_loader):
        """用于验证模型在验证集上的表现"""
        self.model.eval()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():  # 不需要计算梯度
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels)
                total_preds += labels.size(0)

        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = correct_preds.double() / total_preds
        return epoch_loss, epoch_accuracy

    def train_kfold(self, k=5):
        kfold = KFold(n_splits=k, shuffle=True)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            print(f"Fold {fold + 1}/{k}")

            # 划分训练集和验证集
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32)

            best_accuracy = 0.0  # 用于跟踪最佳模型

            for epoch in range(5):  # 训练 5 个 epoch
                train_loss, train_accuracy = self.train_one_epoch(train_loader)
                print(f"Epoch {epoch + 1}/5 - Loss: {train_loss:.4f} - Accuracy: {train_accuracy:.4f}")

                # 在验证集上进行验证
                val_loss, val_accuracy = self.evaluate(val_loader)
                print(f"Validation - Loss: {val_loss:.4f} - Accuracy: {val_accuracy:.4f}")

                # 如果验证集上的准确率更好，保存模型
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    model_path = f"iris_model_fold{fold + 1}_best.pth"
                    torch.save(self.model.state_dict(), model_path)
                    print(f"Best model saved at fold {fold + 1} with accuracy: {val_accuracy:.4f}")

            fold_results.append((val_loss, val_accuracy))

        avg_accuracy = sum([x[1] for x in fold_results]) / len(fold_results)
        return avg_accuracy
