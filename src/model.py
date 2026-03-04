import torch
import torch.nn as nn

class MinecraftCNN(nn.Module, in_channels=3,
                   num_kernels=1, kernel_size=3, 
                   num_conv_layers=1, num_classes=60):
    def __init__(self):
        super(MinecraftCNN, self).__init__()

    def forward(self, x):
        return x
