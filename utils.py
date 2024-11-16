# utils.py

import torch
import torch.nn as nn
import numpy as np
import struct

label_mapping = {
    0: 'person1',
    1: 'person2',
    2: 'person3'
}

def save_model_parameters(model, filename_prefix='model_params'):
    """
    Extract the model's structure and parameters, saving them in .npy, .bin, and .hex formats.
    """
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.intrinsic.quantized.ConvReLU2d, nn.quantized.Conv2d, nn.quantized.Linear)):
            # For quantized modules, weights are in quantized format
            weight = module.weight().int_repr().cpu().numpy()
            weight_scale = module.weight().q_scale()
            weight_zero_point = module.weight().q_zero_point()

            # Save weights in multiple formats
            weight_filename_npy = f'{filename_prefix}_layer{layer_idx}_{name}_weights.npy'
            weight_filename_bin = f'{filename_prefix}_layer{layer_idx}_{name}_weights.bin'
            weight_filename_hex = f'{filename_prefix}_layer{layer_idx}_{name}_weights.hex'

            np.save(weight_filename_npy, weight)
            weight.astype(np.int8).tofile(weight_filename_bin)  # Save as .bin (int8)

            # Save weights as hex
            with open(weight_filename_hex, 'w') as hex_file:
                hex_values = [format(w & 0xFF, '02x') for w in weight.flatten()]
                hex_file.write(' '.join(hex_values))
            print(f'Saved weights to {weight_filename_npy}, {weight_filename_bin}, {weight_filename_hex}')

            # Bias is stored as float32
            bias_tensor = module.bias()
            if bias_tensor is not None:
                bias = bias_tensor.detach().cpu().numpy()
                bias_filename_npy = f'{filename_prefix}_layer{layer_idx}_{name}_biases.npy'
                bias_filename_bin = f'{filename_prefix}_layer{layer_idx}_{name}_biases.bin'
                bias_filename_hex = f'{filename_prefix}_layer{layer_idx}_{name}_biases.hex'

                np.save(bias_filename_npy, bias)
                bias.astype(np.float32).tofile(bias_filename_bin)  # Save as .bin (float32)

                # Save biases as hex
                with open(bias_filename_hex, 'w') as hex_file:
                    hex_values = [format(struct.unpack('<I', struct.pack('<f', b))[0], '08x') for b in bias.flatten()]
                    hex_file.write(' '.join(hex_values))
                print(f'Saved biases to {bias_filename_npy}, {bias_filename_bin}, {bias_filename_hex}')

            # Save layer configuration
            config_filename = f'{filename_prefix}_layer{layer_idx}_{name}_config.txt'
            with open(config_filename, 'w') as f:
                if isinstance(module, (nn.intrinsic.quantized.ConvReLU2d, nn.quantized.Conv2d)):
                    f.write(f'Layer Type: Quantized Conv2d\n')
                    f.write(f'In_channels: {module.in_channels}\n')
                    f.write(f'Out_channels: {module.out_channels}\n')
                    f.write(f'Kernel_size: {module.kernel_size}\n')
                    f.write(f'Stride: {module.stride}\n')
                    f.write(f'Padding: {module.padding}\n')
                    f.write(f'Dilation: {module.dilation}\n')
                    f.write(f'Groups: {module.groups}\n')
                    f.write(f'Weight Scale: {weight_scale}\n')
                    f.write(f'Weight Zero Point: {weight_zero_point}\n')
                elif isinstance(module, nn.quantized.Linear):
                    f.write(f'Layer Type: Quantized Linear\n')
                    f.write(f'In_features: {module.in_features}\n')
                    f.write(f'Out_features: {module.out_features}\n')
                    f.write(f'Weight Scale: {weight_scale}\n')
                    f.write(f'Weight Zero Point: {weight_zero_point}\n')
            print(f'Saved layer configuration to {config_filename}')

            layer_idx += 1
