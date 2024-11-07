# visualize_quantized_model.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import IrisNet
from quantize import fuse_model  # 导入 fuse_model 函数
from torch.quantization import default_qconfig, prepare, convert

def print_model_structure(model):
    """
    打印模型的结构和参数信息。
    """
    print("Quantized Model Structure:\n")
    for name, module in model.named_modules():
        print(f"{name}: {module}")

def visualize_weights(model, filename_prefix='quantized_model'):
    """
    可视化量化模型的权重分布。
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.intrinsic.quantized.ConvReLU2d, nn.quantized.Conv2d, nn.quantized.Linear)):
            # 获取量化后的权重
            weight = module.weight().int_repr().cpu().numpy().flatten()
            weight_scale = module.weight().q_scale()
            weight_zero_point = module.weight().q_zero_point()

            # 将量化后的权重值转换为实际值
            weight_float = (weight.astype(np.float32) - weight_zero_point) * weight_scale

            # 绘制权重直方图
            plt.figure()
            plt.hist(weight_float, bins=100, color='blue', alpha=0.7)
            plt.title(f'Weight Distribution of {name}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(f'{filename_prefix}_{name}_weights.png')
            plt.close()
            print(f'Weight distribution for {name} saved as {filename_prefix}_{name}_weights.png')

def main():
    # 创建模型实例
    model = IrisNet(num_classes=3)
    
    # 模型量化步骤
    # 将模型移动到 CPU
    model.to('cpu')
    model.eval()
    
    # 融合模型模块
    fuse_model(model)
    
    # 指定量化配置
    model.qconfig = default_qconfig
    
    # 准备模型进行静态量化
    prepare(model, inplace=True)
    
    # 重要：在转换模型之前，使用一些输入数据执行前向传播，以收集校准信息
    # 创建一个随机输入，模拟实际数据
    example_input = torch.randn(1, 1, 224, 224)  # 根据您的模型输入尺寸调整
    model(example_input)
    
    # 转换模型为量化版本
    convert(model, inplace=True)
    
    # 加载量化后的模型状态字典
    model.load_state_dict(torch.load('quantized_iris_model.pth'))
    model.eval()
    
    # 打印模型结构
    print_model_structure(model)
    
    # 可视化权重分布
    visualize_weights(model)

if __name__ == '__main__':
    main()
