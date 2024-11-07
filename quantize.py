import torch
import torch.nn as nn
from torch.quantization import prepare, convert, fuse_modules, default_qconfig

def fuse_model(model):
    """
    Fuse Conv, BN, and ReLU modules in the model for quantization.
    """
    for module_name, module in model.named_children():
        if module_name == 'features':
            for idx in range(len(module)):
                # Identify patterns of Conv2d -> BatchNorm2d -> ReLU
                if isinstance(module[idx], nn.Conv2d):
                    fuse_list = []
                    fuse_list.append(f'features.{idx}')
                    if idx + 1 < len(module) and isinstance(module[idx + 1], nn.BatchNorm2d):
                        fuse_list.append(f'features.{idx + 1}')
                        if idx + 2 < len(module) and isinstance(module[idx + 2], nn.ReLU):
                            fuse_list.append(f'features.{idx + 2}')
                    if len(fuse_list) > 1:
                        fuse_modules(model, fuse_list, inplace=True)
    return model

def quantize_model(model, data_loader):
    """
    Statistically quantize the model using PyTorch's quantization utilities.
    """
    # Move model to CPU
    model.to('cpu')

    # Set the model to evaluation mode
    model.eval()

    # Fuse modules
    model_fused = fuse_model(model)

    # Specify quantization configuration
    model_fused.qconfig = default_qconfig

    # Prepare the model for static quantization
    prepare(model_fused, inplace=True)

    # Calibrate the model with a few batches
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to('cpu')  # Ensure images are on CPU
            model_fused(images)
            break  # Only need a few batches for calibration

    # Convert the model to a quantized version
    quantized_model = convert(model_fused, inplace=False)

    return quantized_model
