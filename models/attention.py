# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class PAM(nn.Module):
    def __init__(self, inp, oup, reduction=32, input_size=(9, 9)):
        super(PAM, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.pool_half = nn.AdaptiveAvgPool2d((input_size[0] // 2, input_size[1] // 2))
        self.conv_half = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x)
        # c*H*1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        # half pool
        x_half = self.pool_half(x).reshape(n, c, -1, 1)
        y = torch.cat([x_h, x_w, x_half], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w, x_half = torch.split(y, [h, w, (h // 2) ** 2], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_half = x_half.reshape(n, -1, h // 2, w // 2)
        x_half = F.interpolate(x_half, size=(h, w), mode='nearest')
        x_half = self.conv_half(x_half).sigmoid()
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h * x_half
        return out
