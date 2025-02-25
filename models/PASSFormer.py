# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchinfo import summary
from thop import profile

from models.PConv import PConvBlock
from models.attention import PAM


class LayerNorm(nn.Module):
    """
    From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6,
                 data_format="channels_first"):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_first", "channels_last"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape,
                                self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(dim=1, keepdim=True)
            s = (x - u).pow(2).mean(dim=1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1,
                             groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, shortcut=False, expansion=4):
        super().__init__()
        self.shortcut = shortcut
        hidden_dim = int(in_channels * expansion)

        self.attn = PAM(out_channels, out_channels, input_size=image_size)
        self.conv = PConvBlock(in_channels, hidden_dim, out_channels)

        if self.shortcut:
            self.proj = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1, stride=1,
                                  padding=0, bias=False)

        self.norm = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        short_cut = x
        x1 = self.conv(x)
        x1 = self.attn(x1)
        x = self.proj(short_cut) + x1
        x = self.gelu(self.norm(x))
        return x


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, image_size,
                 heads=4, dim_head=64, dropout=0.):
        super().__init__()
        project_out = not (heads == 1 and dim_head == in_channels)

        self.ih, self.iw = image_size
        head_dim = in_channels // heads
        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.pos_embed = nn.Parameter(torch.zeros(1, self.ih * self.iw, (self.ih + self.iw + (self.ih // 2) ** 2)))
        nn.init.trunc_normal_(self.pos_embed, std=0.2)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = rearrange(q, "b (h d) ih iw -> b h (ih iw) d", h=self.heads)

        k_h = F.adaptive_avg_pool2d(k, (None, 1))
        k_w = F.adaptive_avg_pool2d(k, (1, None)).permute(0, 1, 3, 2)
        k_p = F.adaptive_avg_pool2d(k, (k.shape[2] // 2, k.shape[3] // 2)).reshape(k.shape[0], k.shape[1], -1, 1)
        k = torch.cat([k_h, k_w, k_p], dim=2)

        k = k.sigmoid()
        k = k.squeeze(-1).permute(0, 2, 1)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)

        v_h = F.adaptive_avg_pool2d(v, (None, 1))
        v_w = F.adaptive_avg_pool2d(v, (1, None)).permute(0, 1, 3, 2)
        v_p = F.adaptive_avg_pool2d(v, (v.shape[2] // 2, v.shape[3] // 2)).reshape(v.shape[0], v.shape[1], -1, 1)
        v = torch.cat([v_h, v_w, v_p], dim=2)

        v = v.sigmoid()
        v = v.squeeze(-1).permute(0, 2, 1)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots + self.pos_embed

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h (ih iw) d -> b (h d) ih iw", ih=self.ih, iw=self.iw)
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, in_channels, out_channels, image_size,
                 heads=2, dim_head=64, shortcut=False, dropout=0.):
        super().__init__()

        self.ih, self.iw = image_size
        self.shortcut = shortcut

        if self.shortcut:
            self.proj = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1, stride=1,
                                  padding=0, bias=False)

        self.attn = nn.Sequential(
            LayerNorm(in_channels, data_format="channels_first"),
            Attention(in_channels, out_channels, image_size, heads, dim_head, dropout)
        )
        self.ff = MLP(dim=out_channels)

    def forward(self, x):
        if self.shortcut:
            x = self.proj(x) + self.attn(x)
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class PASSNet(nn.Module):
    def __init__(self, in_channels,
                 channels=(64, 128, 64, 64),
                 image_size=(9, 9),
                 num_classes=16):
        super().__init__()
        ih, iw = image_size
        self.s1 = MBConv(in_channels, channels[0], image_size, shortcut=True)
        self.s2 = MBConv(channels[0], channels[1], image_size, shortcut=True)
        self.s3 = Transformer(channels[1], channels[2], image_size, shortcut=True)
        self.s4 = Transformer(channels[2], channels[3], image_size, shortcut=False)

        self.pool = nn.AvgPool2d(ih, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x


if __name__ == '__main__':
    band, height, width = 30, 9, 9
    img = torch.randn(1, band, height, width)
    net = PASSNet(in_channels=band,
                  channels=(64, 128, 64, 64),
                  image_size=(9, 9),
                  num_classes=9)
    out = summary(net, (1, band, height, width), device='cpu', depth=5)
    # print(out)
    flops, params = profile(net, inputs=(img,))
    print('FLOPs = ' + str(flops / 1000 ** 2) + 'M')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
