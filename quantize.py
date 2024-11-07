# quantize.py

import torch
import torch.nn as nn
from torch.quantization import prepare, convert, fuse_modules, default_qconfig

def fuse_model(model):
    """
    Fuse Conv, BN, and ReLU modules in the model for quantization.
    """
    # Fuse modules in the features sequential
    modules_to_fuse = []
    for idx in range(len(model.features)):
        if isinstance(model.features[idx], nn.Conv2d):
            # Check if next layers are BatchNorm and ReLU
            fuse = [f'features.{idx}']
            if idx + 1 < len(model.features) and isinstance(model.features[idx + 1], nn.BatchNorm2d):
                fuse.append(f'features.{idx + 1}')
                if idx + 2 < len(model.features) and isinstance(model.features[idx + 2], nn.ReLU):
                    fuse.append(f'features.{idx + 2}')
            if len(fuse) > 1:
                fuse_modules(model, fuse, inplace=True)

    # Fuse modules in the classifier sequential
    fuse_modules(model, [['classifier.0', 'classifier.1']], inplace=True)

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
