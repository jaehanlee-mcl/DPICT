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

class subpel_conv_slide(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels_starts, in_channels_ends, out_channels_starts, out_channels_ends,
                 upscale_factor, kernel_size=3, padding=1):
        super(subpel_conv_slide, self).__init__()

        self.num_slides = len(in_channels_starts)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels_starts = in_channels_starts
        self.in_channels_ends = in_channels_ends
        self.out_channels_starts = out_channels_starts
        self.out_channels_ends = out_channels_ends

        self.mid_channels = self.out_channels * upscale_factor ** 2
        self.mid_channels_starts = []
        self.mid_channels_ends = []
        for index_list in range(self.num_slides):
            self.mid_channels_starts.append(self.out_channels_starts[index_list] * upscale_factor ** 2)
            self.mid_channels_ends.append(self.out_channels_ends[index_list] * upscale_factor ** 2)

        self.conv = conv_slide(self.in_channels, self.mid_channels, self.in_channels_starts, self.in_channels_ends, self.mid_channels_starts, self.mid_channels_ends,
                               kernel_size=kernel_size, padding=padding)
        self.shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, input, index_channel=0):
        out = self.conv(input, index_channel=index_channel)
        out = self.shuffle(out)
        return out

class conv_slide(nn.Conv2d):
    def __init__(self, in_channels, out_channels, in_channels_starts, in_channels_ends, out_channels_starts, out_channels_ends,
                 kernel_size=3, stride=1, padding=1, dilation=1, groups_list=[1], bias=True):
        super(conv_slide, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=max(groups_list), bias=bias)

        self.num_slides = len(in_channels_starts)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels_starts = in_channels_starts
        self.in_channels_ends = in_channels_ends
        self.out_channels_starts = out_channels_starts
        self.out_channels_ends = out_channels_ends

        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(self.num_slides)]

    def forward(self, input, index_channel=0):
        in_start = self.in_channels_starts[index_channel]
        in_end = self.in_channels_ends[index_channel]
        out_start = self.out_channels_starts[index_channel]
        out_end = self.out_channels_ends[index_channel]
        groups = self.groups_list[index_channel]

        weight = self.weight[out_start:out_end, in_start:in_end, :, :]
        if self.bias is not None:
            bias = self.bias[out_start:out_end]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, groups)
        return y

class masked_conv_slide(nn.Conv2d):

    def __init__(self, in_channels, out_channels, in_channels_starts, in_channels_ends, out_channels_starts, out_channels_ends,
                 kernel_size=3, stride=1, padding=1, dilation=1, groups_list=[1], bias=True, mask_type: str = "A"):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=max(groups_list), bias=bias)

        self.num_slides = len(in_channels_starts)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels_starts = in_channels_starts
        self.in_channels_ends = in_channels_ends
        self.out_channels_starts = out_channels_starts
        self.out_channels_ends = out_channels_ends

        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(self.num_slides)]

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, input, index_channel=0):
        in_start = self.in_channels_starts[index_channel]
        in_end = self.in_channels_ends[index_channel]
        out_start = self.out_channels_starts[index_channel]
        out_end = self.out_channels_ends[index_channel]
        groups = self.groups_list[index_channel]

        self.weight.data *= self.mask
        weight = self.weight[out_start:out_end, in_start:in_end, :, :]
        if self.bias is not None:
            bias = self.bias[out_start:out_end]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, groups)
        return y

class ResidualBlockWithStrideSlide(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, in_ch_start, in_ch_end, out_ch_start, out_ch_end, stride: int = 2):
        super().__init__()

        norm_ch = []
        for index in range(len(out_ch_end)):
            norm_ch.append(out_ch_end[index] - out_ch_start[index])

        self.conv1 = conv_slide(in_ch, out_ch, in_ch_start, in_ch_end, out_ch_start, out_ch_end, kernel_size=3, stride=stride, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv_slide(out_ch, out_ch, out_ch_start, out_ch_end, out_ch_start, out_ch_end, kernel_size=3, padding=1)
        self.gdn = SwitchableGDN2d(norm_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv_slide(in_ch, out_ch, in_ch_start, in_ch_end, out_ch_start, out_ch_end, kernel_size=1, stride=stride, padding=0)
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


class ResidualBlockUpsampleSlide(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, in_ch_start, in_ch_end, out_ch_start, out_ch_end, upsample: int = 2):
        super().__init__()

        norm_ch = []
        for index in range(len(out_ch_end)):
            norm_ch.append(out_ch_end[index] - out_ch_start[index])

        self.subpel_conv = subpel_conv_slide(in_ch, out_ch, in_ch_start, in_ch_end, out_ch_start, out_ch_end, upscale_factor=upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv_slide(out_ch, out_ch, out_ch_start, out_ch_end, out_ch_start, out_ch_end, kernel_size=3, padding=1)
        self.igdn = SwitchableGDN2d(norm_ch, inverse=True)
        self.subpel = subpel_conv_slide(in_ch, out_ch, in_ch_start, in_ch_end, out_ch_start, out_ch_end, upscale_factor=upsample)

    def forward(self, x, index_channel=0):
        identity = x
        out = self.subpel_conv(x, index_channel)
        out = self.leaky_relu(out)
        out = self.conv(out, index_channel)
        out = self.igdn(out, index_channel)
        identity = self.subpel(identity, index_channel)
        out += identity
        return out

class ResidualBlockSlide(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch, in_ch_start, in_ch_end, out_ch_start, out_ch_end):
        super().__init__()
        self.conv1 = conv_slide(in_ch, out_ch, in_ch_start, in_ch_end, out_ch_start, out_ch_end, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv_slide(out_ch, out_ch, out_ch_start, out_ch_end, out_ch_start, out_ch_end, kernel_size=3, padding=1)
        if in_ch != out_ch:
            self.skip = conv_slide(in_ch, out_ch, in_ch_start, in_ch_end, out_ch_start, out_ch_end, kernel_size=1, padding=0)
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

class AttentionBlockSlide(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, io_ch, mid_ch, io_ch_start, io_ch_end, mid_ch_start, mid_ch_end):
        super().__init__()

        self.res_a1 = ResidualUnit(io_ch=io_ch, mid_ch=mid_ch, io_ch_start=io_ch_start, io_ch_end=io_ch_end, mid_ch_start=mid_ch_start, mid_ch_end=mid_ch_end)
        self.res_a2 = ResidualUnit(io_ch=io_ch, mid_ch=mid_ch, io_ch_start=io_ch_start, io_ch_end=io_ch_end, mid_ch_start=mid_ch_start, mid_ch_end=mid_ch_end)
        self.res_a3 = ResidualUnit(io_ch=io_ch, mid_ch=mid_ch, io_ch_start=io_ch_start, io_ch_end=io_ch_end, mid_ch_start=mid_ch_start, mid_ch_end=mid_ch_end)

        self.res_b1 = ResidualUnit(io_ch=io_ch, mid_ch=mid_ch, io_ch_start=io_ch_start, io_ch_end=io_ch_end, mid_ch_start=mid_ch_start, mid_ch_end=mid_ch_end)
        self.res_b2 = ResidualUnit(io_ch=io_ch, mid_ch=mid_ch, io_ch_start=io_ch_start, io_ch_end=io_ch_end, mid_ch_start=mid_ch_start, mid_ch_end=mid_ch_end)
        self.res_b3 = ResidualUnit(io_ch=io_ch, mid_ch=mid_ch, io_ch_start=io_ch_start, io_ch_end=io_ch_end, mid_ch_start=mid_ch_start, mid_ch_end=mid_ch_end)
        self.conv_b4 = conv_slide(io_ch, io_ch, io_ch_start, io_ch_end, io_ch_start, io_ch_end, kernel_size=1, padding=0)

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

    def __init__(self, io_ch, mid_ch, io_ch_start, io_ch_end, mid_ch_start, mid_ch_end):
        super().__init__()
        self.conv1 = conv_slide(io_ch, mid_ch, io_ch_start, io_ch_end, mid_ch_start, mid_ch_end, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv_slide(mid_ch, mid_ch, mid_ch_start, mid_ch_end, mid_ch_start, mid_ch_end, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv_slide(mid_ch, io_ch, mid_ch_start, mid_ch_end, io_ch_start, io_ch_end, kernel_size=1, padding=0)
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
