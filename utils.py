# utils.py

import numpy as np

def save_model_parameters(model, filename_prefix='model_params'):
    """
    Extract the model's structure and parameters, saving them in .npy, .bin, and .hex formats.
    """
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.quantized.Conv2d, nn.quantized.Linear, nn.intrinsic.quantized.ConvReLU2d)):
            # For quantized modules, weights are in quantized format
            weight = module.weight().int_repr().cpu().numpy()
            weight_scale = module.weight().q_scale()
            weight_zero_point = module.weight().q_zero_point()

            # Save weights in multiple formats
            weight_filename_npy = f'{filename_prefix}_layer{layer_idx}_{name}_weights.npy'
            weight_filename_bin = f'{filename_prefix}_layer{layer_idx}_{name}_weights.bin'
            weight_filename_hex = f'{filename_prefix}_layer{layer_idx}_{name}_weights.hex'
            
            np.save(weight_filename_npy, weight)
            weight.tofile(weight_filename_bin)  # Save as .bin
            with open(weight_filename_hex, 'w') as hex_file:
                hex_file.write(' '.join(format(w, '04x') for w in weight.flatten()))  # Save as .hex

            print(f'Saved weights to {weight_filename_npy}, {weight_filename_bin}, {weight_filename_hex}')

            # Bias is stored as float32
            if module.bias is not None:
                bias = module.bias.detach().cpu().numpy()
                bias_filename_npy = f'{filename_prefix}_layer{layer_idx}_{name}_biases.npy'
                bias_filename_bin = f'{filename_prefix}_layer{layer_idx}_{name}_biases.bin'
                bias_filename_hex = f'{filename_prefix}_layer{layer_idx}_{name}_biases.hex'

                np.save(bias_filename_npy, bias)
                bias.tofile(bias_filename_bin)  # Save as .bin
                with open(bias_filename_hex, 'w') as hex_file:
                    hex_file.write(' '.join(format(int(b), '08x') for b in bias.flatten()))  # Save as .hex
                
                print(f'Saved biases to {bias_filename_npy}, {bias_filename_bin}, {bias_filename_hex}')

            # Save layer configuration
            config_filename = f'{filename_prefix}_layer{layer_idx}_{name}_config.txt'
            with open(config_filename, 'w') as f:
                if isinstance(module, (nn.quantized.Conv2d, nn.intrinsic.quantized.ConvReLU2d)):
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
