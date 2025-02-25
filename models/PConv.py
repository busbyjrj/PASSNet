# !/usr/bin/env python
# -*-coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torchinfo import summary


class PConv(nn.Module):
    """
    Refer Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks
    Code from https://github.com/liaomingg/FasterNet
    """

    def __init__(self,
                 dim: int,
                 n_div: int,
                 forward: str = "split_cat",
                 kernel_size: int = 3,
                 reverse: bool = False,
                 ) -> None:
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.reverse = reverse

        self.conv = nn.Conv2d(self.dim_conv,
                              self.dim_conv,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=(kernel_size - 1) // 2,
                              bias=False)

        if forward == "split_cat":
            self.forward = self._forward_split_cat
        elif forward == "splicing":
            self.forward = self._forward_splicing
        else:
            raise NotImplementedError

    def _forward_splicing(self, x: tensor) -> tensor:
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
        return x

    def _forward_split_cat(self, x: tensor) -> tensor:
        if self.reverse:
            x2, x1 = torch.split(x, [self.dim_untouched, self.dim_conv], dim=1)
        else:
            x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x = torch.cat([x1, x2], dim=1)
        return x


class FasterBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 ):
        expand_ratio = 2

        super().__init__()
        self.conv = nn.Sequential(
            PConv(in_channels, 4, forward="split_cat"),
            nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.GELU(),
            nn.Conv2d(in_channels * expand_ratio, in_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
        )

    def forward(self, x: tensor) -> tensor:
        x = x + self.conv(x)
        return x


def PConvBlock(in_channels, hidden_dim, out_channels):
    conv = nn.Sequential(
        PConv(dim=in_channels, n_div=2, forward="split_cat"),
        nn.BatchNorm2d(in_channels),
        nn.GELU(),
        # pw-linear
        nn.Conv2d(in_channels, hidden_dim, kernel_size=1,
                  stride=1, padding=0,
                  groups=1, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        # pw-linear
        nn.Conv2d(hidden_dim, out_channels, kernel_size=1,
                  stride=1, padding=0,
                  groups=1, bias=False),
    )
    return conv
