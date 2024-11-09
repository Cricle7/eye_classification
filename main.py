from train import IrisTrainer
from quantize import quantize_model
from utils import save_model_parameters
from tqdm import tqdm
import torch

def evaluate_quantized_model(model, data_loader, threshold=0.5):
    """
    Evaluate the quantized model on the test set.
    Inputs below the confidence threshold are classified as 'unknown'.
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
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)

            # 判定为已知类别或未知类别
            predicted_classes = predicted.clone()
            predicted_classes[max_probs < threshold] = 3  # 3为“未知”类别标签

            # 计算准确率
            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    print(f'Quantized Model Test Accuracy with Unknown Detection: {test_acc:.4f}')
    return test_acc

def main():
    # Initialize the trainer
    trainer = IrisTrainer(
        dataset_root='iris_dataset',
        num_classes=3,  # 保持为3类
        batch_size=64,
        num_epochs=5,
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

    # Evaluate quantized model with unknown detection
    evaluate_quantized_model(quantized_model, trainer.test_loader, threshold=0.5)

    # Save model parameters for FPGA deployment
    save_model_parameters(quantized_model, filename_prefix='iris_model')

if __name__ == "__main__":
    main()
