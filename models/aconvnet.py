"""A-ConvNet from Chen et al., IEEE TGRS 2016 (SOC baseline)."""

from __future__ import annotations

import torch
from torch import nn


def _init_conv(module: nn.Conv2d, *, last: bool = False) -> None:
    if last:
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.1)


class AConvNet(nn.Module):
    """
    Five convolution layers, three 2x2 max-pool layers, no fully-connected layers.
    Input: (N, C, 88, 88) with C=2 (magnitude + phase) for the reference SOC setup.
    Output: (N, num_classes) logits.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=6, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout)

        for layer in (self.conv1, self.conv2, self.conv3, self.conv4):
            _init_conv(layer)
        _init_conv(self.conv5, last=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.conv5(x)
        return x.flatten(1)
