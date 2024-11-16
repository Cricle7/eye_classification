import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets import IrisDataset
from models import IrisNet

class IrisTrainer:
    def __init__(self, dataset_root='iris_dataset', num_classes=3, batch_size=64, num_epochs=50, learning_rate=0.0005):
        self.dataset_root = dataset_root
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((40, 40)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Load dataset
        self.dataset = IrisDataset(root_dir=self.dataset_root, transform=self.transform)
        self.train_loader, self.test_loader = self._prepare_data()

        # Define model, loss function, and optimizer
        self.model = IrisNet(num_classes=self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def _prepare_data(self):
        """
        Use train_test_split to split the dataset and create data loaders.
        """
        # Get indices and labels
        indices = list(range(len(self.dataset)))
        labels = [self.dataset.labels[i] for i in indices]

        # Stratified split
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=42
        )

        # Create datasets
        train_dataset = Subset(self.dataset, train_indices)
        test_dataset = Subset(self.dataset, test_indices)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_loader, test_loader

    def train(self):
        """
        Train the model.
        """
        print(f'Using device: {self.device}')

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = correct / total
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

            # Save model every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f'iris_model_epoch_{epoch+1}.pth')

        # Save final model
        self.save_model('final_iris_model.pth')

    def evaluate(self):
        """
        Evaluate the model on the test set.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc='Evaluating')
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = correct / total
        print(f'Test Accuracy: {test_acc:.4f}')
        return test_acc

    def save_model(self, path='iris_model.pth'):
        """
        Save model parameters.
        """
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path='iris_model.pth'):
        """
        Load model parameters.
        """
        self.model.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')
