"""
Wrapper utilities for MODNet architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IBNorm(nn.Module):
    """Instance-Batch Normalization block."""
    def __init__(self, in_channels):
        super().__init__()
        self.half = in_channels // 2
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(in_channels - self.half)

    def forward(self, x):
        split = torch.split(x, [self.half, x.size(1) - self.half], dim=1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat((out1, out2), 1)

class Conv2dIBNormRelu(nn.Module):
    """Conv2d + (IBNorm or BatchNorm) + LeakyReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, with_ibn=True, with_relu=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias)
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))
        else:
            layers.append(nn.BatchNorm2d(out_channels))

        if with_relu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GaussianBlurLayer(nn.Module):
    """Apply Gaussian blur to a tensor (used for smoothing trimaps)."""
    def __init__(self, channels, kernel_size=15):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert kernel_size % 2 != 0, "Kernel size must be odd"

        # Create Gaussian kernel
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        kernel = torch.zeros(kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            kernel[i] = -(i - center) ** 2 / (2 * sigma ** 2)
        kernel = torch.exp(kernel)
        kernel = kernel / kernel.sum()

        # 2D kernel from outer product
        kernel_2d = kernel.unsqueeze(1) @ kernel.unsqueeze(0)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)

        self.weight = nn.Parameter(data=kernel_2d, requires_grad=False)
        self.padding = kernel_size // 2

    def forward(self, x):
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)
