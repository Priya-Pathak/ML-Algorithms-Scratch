import torch
import torch.nn as nn

# Example input
batch_size = 2
in_channels = 4
height = 8
width = 8

# 1. Input data
x = torch.randn(batch_size, in_channels, height, width)  # Shape: [2, 4, 8, 8]

# 2. Depthwise convolution
# To do depthwise: out_channels == in_channels, groups == in_channels
kernel_size = 3
padding = 1  # To keep output same spatial size as input
depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                           kernel_size=kernel_size, groups=in_channels, padding=padding)

# 3. Forward pass
y = depthwise_conv(x)  # Shape: [2, 4, 8, 8]

print("Input shape (x):", x.shape)
print("Weight shape:", depthwise_conv.weight.shape)
print("Output shape (y):", y.shape)
