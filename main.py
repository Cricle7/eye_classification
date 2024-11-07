# main.py

from train import IrisTrainer
from quantize import quantize_model
from utils import save_model_parameters
from tqdm import tqdm
import torch

def evaluate_quantized_model(model, data_loader):
    """
    Evaluate the quantized model on the test set.
    """
    device = torch.device('cpu')  # Ensure evaluation is on CPU
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Evaluating Quantized Model')
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total
    print(f'Quantized Model Test Accuracy: {test_acc:.4f}')
    return test_acc

def main():
    # Initialize the trainer
    trainer = IrisTrainer(
        dataset_root='iris_dataset',
        num_classes=3,
        batch_size=64,
        num_epochs=50,
        learning_rate=0.0005
    )

    # Train the model
    trainer.train()

    # Evaluate the trained model on the test set
    trainer.evaluate()

    # Quantize the model
    quantized_model = quantize_model(trainer.model, trainer.train_loader)

    # Save quantized model
    torch.save(quantized_model.state_dict(), 'quantized_iris_model.pth')
    print('Quantized model saved to quantized_iris_model.pth')

    # Evaluate quantized model
    evaluate_quantized_model(quantized_model, trainer.test_loader)

    # Save model parameters for FPGA deployment
    save_model_parameters(quantized_model, filename_prefix='iris_model')

if __name__ == "__main__":
    main()
