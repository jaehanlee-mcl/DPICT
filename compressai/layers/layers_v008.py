# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .layers import *
from typing import Any

import torch
import torch.nn as nn
import math

from torch import Tensor

from .gdn import GDN
WIDTH_MULT_LIST = [0.25, 0.50, 0.75, 1.00]

class SwitchableGDN2d(nn.Module):
    def __init__(self, num_features_list, inverse: bool = False):
        super(SwitchableGDN2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        self.inverse = bool(inverse)

        gdns = []
        for i in num_features_list:
            gdns.append(GDN(i, inverse=self.inverse))
        self.gdn = nn.ModuleList(gdns)
        self.ignore_model_profiling = True

    def forward(self, input, index_channel=0):
        y = self.gdn[index_channel](input)
        return y


class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(WIDTH_MULT_LIST)
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = WIDTH_MULT_LIST.index(self.width_mult)
        y = self.bn[idx](input)
        return y

class subpel_conv_slim(nn.Module):
    def __init__(self, in_channels_list, out_channels_list, upscale_factor, kernel_size=3, padding=1):
        super(subpel_conv_slim, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.mid_channels_list = []
        for index_list in range(len(out_channels_list)):
            self.mid_channels_list.append(self.out_channels_list[index_list] * upscale_factor ** 2)

        self.conv = conv_slim(self.in_channels_list, self.mid_channels_list, kernel_size=kernel_size, padding=padding)
        self.shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, input, index_channel=0):
        out = self.conv(input, index_channel=index_channel)
        out = self.shuffle(out)
        return out

class conv_slim(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list, kernel_size=3, stride=1, padding=1, dilation=1, groups_list=[1], bias=True):
        super(conv_slim, self).__init__(max(in_channels_list), max(out_channels_list), kernel_size, stride=stride, padding=padding, dilation=dilation, groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]

    def forward(self, input, index_channel=0):
        self.in_channels = self.in_channels_list[index_channel]
        self.out_channels = self.out_channels_list[index_channel]
        self.groups = self.groups_list[index_channel]

        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return y

class masked_conv_slim(nn.Conv2d):

    def __init__(self, in_channels_list, out_channels_list, kernel_size=3, stride=1, padding=1, dilation=1, groups_list=[1], bias=True, mask_type: str = "A"):
        super().__init__(max(in_channels_list), max(out_channels_list), kernel_size, stride=stride, padding=padding, dilation=dilation, groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, input, index_channel=0):
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask

        self.in_channels = self.in_channels_list[index_channel]
        self.out_channels = self.out_channels_list[index_channel]
        self.groups = self.groups_list[index_channel]

        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return y

class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(WIDTH_MULT_LIST)

    def forward(self, input):
        idx = WIDTH_MULT_LIST.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(WIDTH_MULT_LIST)

    def forward(self, input):
        idx = WIDTH_MULT_LIST.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0)

class ResidualBlockWithStrideSlim(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, stride: int = 2):
        super().__init__()
        self.conv1 = conv_slim(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv_slim(out_ch, out_ch, kernel_size=3, padding=1)
        self.gdn = SwitchableGDN2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv_slim(in_ch, out_ch, kernel_size=1, stride=stride, padding=0)
        else:
            self.skip = None

    def forward(self, x, index_channel=0):
        identity = x
        out = self.conv1(x, index_channel)
        out = self.leaky_relu(out)
        out = self.conv2(out, index_channel)
        out = self.gdn(out, index_channel)

        if self.skip is not None:
            identity = self.skip(x, index_channel)

        out += identity
        return out


class ResidualBlockUpsampleSlim(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv_slim(in_ch, out_ch, upscale_factor=upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv_slim(out_ch, out_ch, kernel_size=3, padding=1)
        self.igdn = SwitchableGDN2d(out_ch, inverse=True)
        self.upsample = subpel_conv_slim(in_ch, out_ch, upscale_factor=upsample)

    def forward(self, x, index_channel=0):
        identity = x
        out = self.subpel_conv(x, index_channel)
        out = self.leaky_relu(out)
        out = self.conv(out, index_channel)
        out = self.igdn(out, index_channel)
        identity = self.upsample(identity, index_channel)
        out += identity
        return out

class ResidualBlockSlim(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv_slim(in_ch, out_ch, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv_slim(out_ch, out_ch, kernel_size=3, padding=1)
        if in_ch != out_ch:
            self.skip = conv_slim(in_ch, out_ch, kernel_size=1, padding=0)
        else:
            self.skip = None

    def forward(self, x, index_channel=0):
        identity = x

        out = self.conv1(x, index_channel)
        out = self.leaky_relu(out)
        out = self.conv2(out, index_channel)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x, index_channel)

        out = out + identity
        return out

class AttentionBlockSlim(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, io_ch, mid_ch):
        super().__init__()

        self.res_a1 = ResidualUnit(io_ch=io_ch, mid_ch=mid_ch)
        self.res_a2 = ResidualUnit(io_ch=io_ch, mid_ch=mid_ch)
        self.res_a3 = ResidualUnit(io_ch=io_ch, mid_ch=mid_ch)

        self.res_b1 = ResidualUnit(io_ch=io_ch, mid_ch=mid_ch)
        self.res_b2 = ResidualUnit(io_ch=io_ch, mid_ch=mid_ch)
        self.res_b3 = ResidualUnit(io_ch=io_ch, mid_ch=mid_ch)
        self.conv_b4 = conv_slim(io_ch, io_ch, kernel_size=1, padding=0)

    def forward(self, x, index_channel=0):
        identity = x

        a = x
        a = self.res_a1(a, index_channel)
        a = self.res_a2(a, index_channel)
        a = self.res_a3(a, index_channel)

        b = x
        b = self.res_b1(b, index_channel)
        b = self.res_b2(b, index_channel)
        b = self.res_b3(b, index_channel)
        b = self.conv_b4(b, index_channel)

        out = a * torch.sigmoid(b)
        out += identity
        return out


class ResidualUnit(nn.Module):
    """Simple residual unit."""

    def __init__(self, io_ch, mid_ch):
        super().__init__()
        self.conv1 = conv_slim(io_ch, mid_ch, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv_slim(mid_ch, mid_ch, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv_slim(mid_ch, io_ch, kernel_size=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x, index_channel=0):
        identity = x
        out = x
        out = self.conv1(out, index_channel)
        out = self.relu1(out)
        out = self.conv2(out, index_channel)
        out = self.relu2(out)
        out = self.conv3(out, index_channel)

        out += identity
        out = self.relu3(out)
        return out
