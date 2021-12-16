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

class subpel_conv_divide(nn.Module):
    def __init__(self, in_ch, out_ch,
                 in_shared_ch_start, in_shared_ch_end, in_specific_ch_starts, in_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends,
                 upscale_factor, kernel_size=3, padding=1):
        super(subpel_conv_divide, self).__init__()

        self.num_divides = len(in_specific_ch_starts)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.in_shared_ch_start = in_shared_ch_start
        self.in_shared_ch_end = in_shared_ch_end
        self.in_specific_ch_starts = in_specific_ch_starts
        self.in_specific_ch_ends = in_specific_ch_ends
        self.out_shared_ch_start = out_shared_ch_start
        self.out_shared_ch_end = out_shared_ch_end
        self.out_specific_ch_starts = out_specific_ch_starts
        self.out_specific_ch_ends = out_specific_ch_ends

        self.mid_ch = self.out_ch * upscale_factor ** 2
        self.mid_shared_ch_start = self.out_shared_ch_start * upscale_factor ** 2
        self.mid_shared_ch_end = self.out_shared_ch_end * upscale_factor ** 2
        self.mid_specific_ch_starts = []
        self.mid_specific_ch_ends = []
        for index_list in range(self.num_divides):
            self.mid_specific_ch_starts.append(self.out_specific_ch_starts[index_list] * upscale_factor ** 2)
            self.mid_specific_ch_ends.append(self.out_specific_ch_ends[index_list] * upscale_factor ** 2)

        self.conv = conv_divide(self.in_ch, self.mid_ch,
                                self.in_shared_ch_start, self.in_shared_ch_end, self.in_specific_ch_starts, self.in_specific_ch_ends,
                                self.mid_shared_ch_start, self.mid_shared_ch_end, self.mid_specific_ch_starts, self.mid_specific_ch_ends,
                               kernel_size=kernel_size, padding=padding)
        self.shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, input, index_channel=0):
        out = self.conv(input, index_channel=index_channel)
        out = self.shuffle(out)
        return out

class conv_divide(nn.Conv2d):
    def __init__(self, in_ch, out_ch,
                 in_shared_ch_start, in_shared_ch_end, in_specific_ch_starts, in_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends,
                 kernel_size=3, stride=1, padding=1, dilation=1, groups_list=[1], bias=True):

        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=max(groups_list), bias=bias)

        self.num_divides = len(in_specific_ch_starts)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.in_shared_ch_start = in_shared_ch_start
        self.in_shared_ch_end = in_shared_ch_end
        self.in_specific_ch_starts = in_specific_ch_starts
        self.in_specific_ch_ends = in_specific_ch_ends
        self.out_shared_ch_start = out_shared_ch_start
        self.out_shared_ch_end = out_shared_ch_end
        self.out_specific_ch_starts = out_specific_ch_starts
        self.out_specific_ch_ends = out_specific_ch_ends

        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(self.num_divides)]

    def forward(self, input, index_channel=0):
        in_ch_list \
            = list(range(self.in_shared_ch_start, self.in_shared_ch_end)) \
              + list(range(self.in_specific_ch_starts[index_channel], self.in_specific_ch_ends[index_channel]))
        out_ch_list \
            = list(range(self.out_shared_ch_start, self.out_shared_ch_end)) \
              + list(range(self.out_specific_ch_starts[index_channel], self.out_specific_ch_ends[index_channel]))

        weight = self.weight[out_ch_list, :, :, :][:, in_ch_list, :, :]
        if self.bias is not None:
            bias = self.bias[out_ch_list]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups_list[index_channel])
        return y

class masked_conv_divide(nn.Conv2d):

    def __init__(self, in_ch, out_ch,
                 in_shared_ch_start, in_shared_ch_end, in_specific_ch_starts, in_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends,
                 kernel_size=3, stride=1, padding=1, dilation=1, groups_list=[1], bias=True, mask_type: str = "A"):

        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=max(groups_list), bias=bias)

        self.num_divides = len(in_specific_ch_starts)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.in_shared_ch_start = in_shared_ch_start
        self.in_shared_ch_end = in_shared_ch_end
        self.in_specific_ch_starts = in_specific_ch_starts
        self.in_specific_ch_ends = in_specific_ch_ends
        self.out_shared_ch_start = out_shared_ch_start
        self.out_shared_ch_end = out_shared_ch_end
        self.out_specific_ch_starts = out_specific_ch_starts
        self.out_specific_ch_ends = out_specific_ch_ends

        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(self.num_divides)]

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, input, index_channel=0):
        in_ch_list \
            = list(range(self.in_shared_ch_start, self.in_shared_ch_end)) \
              + list(range(self.in_specific_ch_starts[index_channel], self.in_specific_ch_ends[index_channel]))
        out_ch_list \
            = list(range(self.out_shared_ch_start, self.out_shared_ch_end)) \
              + list(range(self.out_specific_ch_starts[index_channel], self.out_specific_ch_ends[index_channel]))

        self.weight.data *= self.mask
        weight = self.weight[out_ch_list, :, :, :][:, in_ch_list, :, :]
        if self.bias is not None:
            bias = self.bias[out_ch_list]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups_list[index_channel])
        return y

class ResidualBlockWithStrideDivide(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch,
                 in_shared_ch_start, in_shared_ch_end, in_specific_ch_starts, in_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends, stride: int = 2):
        super().__init__()

        norm_ch = []
        for index in range(len(out_specific_ch_starts)):
            num_shared = out_shared_ch_end - out_shared_ch_start
            num_specific = out_specific_ch_ends[index] - out_specific_ch_starts[index]
            norm_ch.append(num_shared + num_specific)

        self.conv1 = conv_divide(in_ch, out_ch,
                 in_shared_ch_start, in_shared_ch_end, in_specific_ch_starts, in_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends, kernel_size=3, stride=stride, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv_divide(out_ch, out_ch,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends, kernel_size=3, padding=1)
        self.gdn = SwitchableGDN2d(norm_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv_divide(in_ch, out_ch,
                 in_shared_ch_start, in_shared_ch_end, in_specific_ch_starts, in_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends, kernel_size=1, stride=stride, padding=0)
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


class ResidualBlockUpsampleDivide(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch,
                 in_shared_ch_start, in_shared_ch_end, in_specific_ch_starts, in_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends, upsample: int = 2):
        super().__init__()

        norm_ch = []
        for index in range(len(out_specific_ch_starts)):
            num_shared = out_shared_ch_end - out_shared_ch_start
            num_specific = out_specific_ch_ends[index] - out_specific_ch_starts[index]
            norm_ch.append(num_shared + num_specific)

        self.subpel_conv = subpel_conv_divide(in_ch, out_ch,
                 in_shared_ch_start, in_shared_ch_end, in_specific_ch_starts, in_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends, upscale_factor=upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv_divide(out_ch, out_ch,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends, kernel_size=3, padding=1)
        self.igdn = SwitchableGDN2d(norm_ch, inverse=True)
        self.subpel = subpel_conv_divide(in_ch, out_ch,
                 in_shared_ch_start, in_shared_ch_end, in_specific_ch_starts, in_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends, upscale_factor=upsample)

    def forward(self, x, index_channel=0):
        identity = x
        out = self.subpel_conv(x, index_channel)
        out = self.leaky_relu(out)
        out = self.conv(out, index_channel)
        out = self.igdn(out, index_channel)
        identity = self.subpel(identity, index_channel)
        out += identity
        return out

class ResidualBlockDivide(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch,
                 in_shared_ch_start, in_shared_ch_end, in_specific_ch_starts, in_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends):
        super().__init__()
        self.conv1 = conv_divide(in_ch, out_ch,
                 in_shared_ch_start, in_shared_ch_end, in_specific_ch_starts, in_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv_divide(out_ch, out_ch,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends, kernel_size=3, padding=1)
        if in_ch != out_ch:
            self.skip = conv_divide(in_ch, out_ch,
                 in_shared_ch_start, in_shared_ch_end, in_specific_ch_starts, in_specific_ch_ends,
                 out_shared_ch_start, out_shared_ch_end, out_specific_ch_starts, out_specific_ch_ends, kernel_size=1, padding=0)
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

class AttentionBlockDivide(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, io_ch, mid_ch,
                 io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends,
                 mid_shared_ch_start, mid_shared_ch_end, mid_specific_ch_starts, mid_specific_ch_ends):
        super().__init__()

        self.res_a1 = ResidualUnit(io_ch, mid_ch,
                 io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends,
                 mid_shared_ch_start, mid_shared_ch_end, mid_specific_ch_starts, mid_specific_ch_ends)
        self.res_a2 = ResidualUnit(io_ch, mid_ch,
                 io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends,
                 mid_shared_ch_start, mid_shared_ch_end, mid_specific_ch_starts, mid_specific_ch_ends)
        self.res_a3 = ResidualUnit(io_ch, mid_ch,
                 io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends,
                 mid_shared_ch_start, mid_shared_ch_end, mid_specific_ch_starts, mid_specific_ch_ends)

        self.res_b1 = ResidualUnit(io_ch, mid_ch,
                 io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends,
                 mid_shared_ch_start, mid_shared_ch_end, mid_specific_ch_starts, mid_specific_ch_ends)
        self.res_b2 = ResidualUnit(io_ch, mid_ch,
                 io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends,
                 mid_shared_ch_start, mid_shared_ch_end, mid_specific_ch_starts, mid_specific_ch_ends)
        self.res_b3 = ResidualUnit(io_ch, mid_ch,
                 io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends,
                 mid_shared_ch_start, mid_shared_ch_end, mid_specific_ch_starts, mid_specific_ch_ends)
        self.conv_b4 = conv_divide(io_ch, io_ch,
                 io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends,
                 io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends, kernel_size=1, padding=0)

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

    def __init__(self, io_ch, mid_ch,
                 io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends,
                 mid_shared_ch_start, mid_shared_ch_end, mid_specific_ch_starts, mid_specific_ch_ends):
        super().__init__()
        self.conv1 = conv_divide(io_ch, mid_ch,
                                 io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends,
                                 mid_shared_ch_start, mid_shared_ch_end, mid_specific_ch_starts, mid_specific_ch_ends, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv_divide(mid_ch, mid_ch,
                                 mid_shared_ch_start, mid_shared_ch_end, mid_specific_ch_starts, mid_specific_ch_ends,
                                 mid_shared_ch_start, mid_shared_ch_end, mid_specific_ch_starts, mid_specific_ch_ends, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv_divide(mid_ch, io_ch,
                                 mid_shared_ch_start, mid_shared_ch_end, mid_specific_ch_starts, mid_specific_ch_ends,
                                 io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends, kernel_size=1, padding=0)
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

def channel_mult(input, coeff):
    output = []
    for index in range(len(input)):
        output.append(math.floor(input[index] * coeff))

    return output