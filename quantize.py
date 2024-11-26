import torch
import torch.nn as nn
from torch.quantization import prepare, convert, fuse_modules, default_qconfig

def fuse_model(model):
    """
    Fuse Conv, BN, and ReLU modules in the model for quantization.
    """
    # Fuse modules in the features sequential
    for idx in range(len(model.features)):
        if isinstance(model.features[idx], nn.Conv2d):
            # 默认情况下，假设 Conv2d 后面跟着 BatchNorm2d 和 ReLU
            modules_to_fuse = []
            if idx + 1 < len(model.features) and isinstance(model.features[idx + 1], nn.BatchNorm2d):
                if idx + 2 < len(model.features) and isinstance(model.features[idx + 2], nn.ReLU):
                    modules_to_fuse = [f'features.{idx}', f'features.{idx + 1}', f'features.{idx + 2}']
                    fuse_modules(model, modules_to_fuse, inplace=True)
                else:
                    modules_to_fuse = [f'features.{idx}', f'features.{idx + 1}']
                    fuse_modules(model, modules_to_fuse, inplace=True)
            elif idx + 1 < len(model.features) and isinstance(model.features[idx + 1], nn.ReLU):
                modules_to_fuse = [f'features.{idx}', f'features.{idx + 1}']
                fuse_modules(model, modules_to_fuse, inplace=True)

    # 如果模型中有其他需要融合的层，可以在这里添加
    # 例如，如果有单独的 Conv2d -> BatchNorm2d -> ReLU 序列
    # 请确保融合的模块名称与模型中的实际模块名称匹配

def quantize_model(model, data_loader):
    """
    Statistically quantize the model using PyTorch's quantization utilities.
    """
    # Move model to CPU
    model.to('cpu')

    # Set the model to evaluation mode
    model.eval()

    # Fuse modules
    fuse_model(model)

    # Specify quantization configuration
    model.qconfig = default_qconfig

    # Prepare the model for static quantization
    prepare(model, inplace=True)

    # Calibrate the model with a few batches
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to('cpu')
            model(images)
            break  # Only need a few batches for calibration

    # Convert the model to a quantized version
    convert(model, inplace=True)

    return model
