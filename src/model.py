import torch
import torch.nn as nn

class MinecraftCNN(nn.Module):
    def __init__(self, in_channels=3, num_kernels=1, kernel_size=3, num_conv_layers=1, num_classes=60):
        super(MinecraftCNN, self).__init__()

    def forward(self, x):
        return x
