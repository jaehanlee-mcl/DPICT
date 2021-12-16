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

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
import numpy as np
import scipy

from scipy.stats import norm
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf

from .utils import conv, deconv, update_registered_buffers
from .priors import get_scale_table

from torchvision.transforms import ToPILImage

from compressai.layers import (
    AttentionBlockDivide,
    ResidualBlockDivide,
    ResidualBlockUpsampleDivide,
    ResidualBlockWithStrideDivide,
    conv_divide,
    subpel_conv_divide,
    masked_conv_divide,
)

from .waseda import Cheng2020Anchor

class DPICT_post_net(Cheng2020Anchor):
    """Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses self-attention, residual blocks with small convolutions (3x3 and 1x1),
    and sub-pixel convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, shared_ratio=[0/1, 1/1], specific_ratios=[1/1, 1/1, 1/1, 1/1], **kwargs):
        super().__init__(N=N, **kwargs)

        num_divide = len(specific_ratios) - 1
        divide_ch = N
        shared_ch_start = round(shared_ratio[0] * divide_ch)
        shared_ch_end = round(shared_ratio[1] * divide_ch)

        specific_ch_starts = []
        specific_ch_ends = []
        for index in range(num_divide):
            specific_ch_starts.append(round(specific_ratios[index] * divide_ch))
            specific_ch_ends.append(round(specific_ratios[index+1] * divide_ch))

        self.g_s = Divide_g_s(divide_ch, shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends)

    def forward(self, x, index_channel=0, quantize_parameters=[0,0,0,0]):
        decoder_index_channel = index_channel
        # decoder_index_channel = 0

        x_hat = self.g_s(x, decoder_index_channel)

        return {
            "x_hat": x_hat,
        }

    def get_quantize_tensor(self, scales_hat, quantize_parameters):
        C, H, W = scales_hat.shape[1], scales_hat.shape[2], scales_hat.shape[3]

        quantize_minimum = np.ones(shape=(C,H,W))

        quantize_floor = math.floor(quantize_parameters[0])
        quantize_ceil = math.ceil(quantize_parameters[0])
        ratio_floor = quantize_ceil - quantize_parameters[0]
        last_floor_index = round(ratio_floor * C)

        quantize_tensor = np.zeros(shape=(C,H,W))
        quantize_tensor[:last_floor_index] = quantize_floor
        quantize_tensor[last_floor_index:] = quantize_ceil

        quantize_tensor = (3 ** quantize_tensor)
        quantize_tensor = np.maximum(quantize_tensor, quantize_minimum)

        return quantize_tensor

    def get_quantize_tensor_eval(self, y, means_hat, scales_hat, quantize_parameters):
        C, H, W = y.shape[1], y.shape[2], y.shape[3]
        quantize_minimum = np.ones(shape=(C,H,W))

        y = y.cpu().detach().numpy()
        means_hat = means_hat.cpu().detach().numpy()
        scales_hat = scales_hat.cpu().detach().numpy()

        print('Quantize parameters: ', str(quantize_parameters[0]))
        quantize_parameter_res = quantize_parameters[0] - np.floor(quantize_parameters[0])
        num_quantize_level = int(np.floor(quantize_parameters[0])+1)
        if quantize_parameter_res == 0:
            quantize_tensor = np.zeros(shape=(C, H, W)) + (num_quantize_level - 1)
        elif quantize_parameter_res > 0:
            y_RDs, y_RD_ranks = self.get_RD_ranks(y, means_hat, scales_hat, num_quantize_level)
            quantize_tensor = np.zeros(shape=(C, H, W)) + (num_quantize_level-1)
            quantize_tensor = quantize_tensor + (y_RD_ranks[:,:,:,:,num_quantize_level-1] <= quantize_parameter_res)

        quantize_tensor = (3 ** quantize_tensor)
        quantize_tensor = np.maximum(quantize_tensor, quantize_minimum)

        return quantize_tensor

    def get_RD_ranks(self, y, means_hat, scales_hat, num_quantize_level=6):

        QI = 3 # quantize interval
        B, C, H, W = y.shape[0], y.shape[1], y.shape[2], y.shape[3]
        rv = scipy.stats.norm(loc=0, scale=1)
        y_res = np.round(y - means_hat)
        valid_num_quantize_level = int(np.ceil(np.log(np.abs(y_res).max()) / np.log(QI)))
        if num_quantize_level <= valid_num_quantize_level:
            y_RDs = np.zeros((B, C, H, W, valid_num_quantize_level))
        else:
            y_RDs = np.zeros((B, C, H, W, num_quantize_level))

        y_bounds = np.arange(pow(QI,valid_num_quantize_level)+1) - pow(QI,valid_num_quantize_level)/2
        y_bounds_sigma \
            = y_bounds[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :] \
              / scales_hat[:, :, :, :, np.newaxis, np.newaxis]
        y_probs = rv.cdf(y_bounds_sigma[:,:,:,:,:,1:]) - rv.cdf(y_bounds_sigma[:,:,:,:,:,:-1])

        if num_quantize_level <= valid_num_quantize_level:
            for index_Q in range(num_quantize_level-1,num_quantize_level):
                min_index = 0
                max_index = pow(QI, valid_num_quantize_level) - 1
                index_y_res = y_res + np.floor(pow(QI, valid_num_quantize_level) / 2)
                index_y_res = index_y_res + (index_y_res<min_index)*(-index_y_res+min_index) + (index_y_res>max_index)*(-index_y_res+max_index)

                scale1_start = np.array(np.floor(index_y_res / pow(QI, index_Q + 1)) * pow(QI, index_Q + 1), dtype='int64')
                scale1_end = scale1_start + pow(QI, index_Q+1)
                scale1_length = int(pow(QI, index_Q + 1))
                scale1_probs = np.zeros((B, C, H, W, 1, scale1_length))
                for index_B in range(B):
                    for index_C in range(C):
                        #print('index_C:', str(index_C))
                        for index_H in range(H):
                            for index_W in range(W):
                                temp_start = scale1_start[index_B,index_C,index_H,index_W]
                                temp_end = scale1_end[index_B,index_C,index_H,index_W]
                                scale1_probs[index_B,index_C,index_H,index_W,0,:] = y_probs[index_B,index_C,index_H,index_W,0,temp_start:temp_end]
                scale1_probs_sum = np.sum(scale1_probs, axis=5, keepdims=True)
                scale1_probs_MSE = self.get_probs_MSE(scale1_probs / (scale1_probs_sum+1e-10))

                delta_rate = np.zeros((B, C, H, W, 1, 1))
                scale0_probs_MSE = np.zeros((B, C, H, W, 1, 1))

                for index_QI in range(QI):
                    temp_start = index_QI * pow(QI, index_Q)
                    temp_end = (index_QI + 1) * pow(QI, index_Q)
                    scale1_probs_part = scale1_probs[:, :, :, :, :, temp_start:temp_end]
                    scale1_probs_part_sum = np.sum(scale1_probs_part, axis=5, keepdims=True)
                    scale1_probs_part_ratio = scale1_probs_part_sum / scale1_probs_sum

                    ## get delta rate
                    temp_delta_rate = - scale1_probs_part_ratio * np.log(scale1_probs_part_ratio + 1e-10)
                    temp_delta_rate = temp_delta_rate + (temp_delta_rate<0)*(-temp_delta_rate)
                    delta_rate = delta_rate + temp_delta_rate

                    ## get delta distortion
                    scale0_probs_MSE = scale0_probs_MSE \
                                       + scale1_probs_part_sum / (scale1_probs_sum+1e-10) \
                                       * self.get_probs_MSE(scale1_probs_part / (scale1_probs_part_sum+1e-10))
                delta_distortion = scale0_probs_MSE - scale1_probs_MSE

                RD = np.squeeze(- delta_distortion / (delta_rate + 1e-10), axis=(4,5)) + 1e-10
                y_RDs[:, :, :, :, index_Q] = RD

        prior = np.power(y_res,2)[:,:,:,:,np.newaxis] + 1e-10
        y_RD_ranks = self.get_ranks(y_RDs, prior, B, C, H, W)

        return y_RDs, y_RD_ranks

    def get_probs_MSE(self, probs):
        indexs = np.arange(probs.shape[5])[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
        probs_mean = np.sum(probs * indexs, axis=5, keepdims=True)
        diffs = indexs - probs_mean
        probs_MSE = np.sum(diffs * diffs * probs, axis=5, keepdims=True)

        return probs_MSE

    def get_ranks(self, data, prior, B, C, H, W):
        data_reshape = (data * prior).reshape(B, C * H * W, 1, 1, data.shape[4])
        data_ranks_reshape = np.zeros(data_reshape.shape)
        arange_rank = np.arange(C * H * W) / (C * H * W)
        for index_Q in range(data.shape[4]):
            data_reshape_rank = np.argsort(data_reshape[0, :, 0, 0, index_Q])
            data_ranks_reshape[0, data_reshape_rank, 0, 0, index_Q] = arange_rank
        data_ranks = data_ranks_reshape.reshape(B, C, H, W, data.shape[4])

        return data_ranks

    @classmethod
    def from_state_dict(cls, state_dict, shared_ratio=[0/8, 5/8], specific_ratios=[5/8, 6/8, 7/8, 8/8]):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.res1.conv1.weight"].size(0)
        net = cls(N=N, shared_ratio=shared_ratio, specific_ratios=specific_ratios)
        net.load_state_dict(state_dict)

        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated1 = self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated2 = False
        for m in self.entropy_bottleneck.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated2 |= rv

        updated1 |= updated2

        return updated1

class Divide_g_s(nn.Module):

    def __init__(self, divide_ch, shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends):
        super().__init__()

        num_divide = len(specific_ch_starts)
        io_ch = 3
        io_shared_ch_start = 0
        io_shared_ch_end = 3
        io_specific_ch_starts = [3] * num_divide
        io_specific_ch_ends = [3] * num_divide

        divide_ch_x0_5 = math.floor(divide_ch * 0.5)
        shared_ch_start_x0_5 = math.floor(shared_ch_start * 0.5)
        shared_ch_end_x0_5 = math.floor(shared_ch_end * 0.5)
        specific_ch_starts_x0_5 = channel_mult(specific_ch_starts, 0.5)
        specific_ch_ends_x0_5 = channel_mult(specific_ch_ends, 0.5)

        self.res1 = ResidualBlockWithStrideDivide(io_ch, divide_ch,
                                                  io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends,
                                                  shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends, stride=2)
        self.res2 = ResidualBlockDivide(divide_ch, divide_ch,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends)
        self.res3 = ResidualBlockWithStrideDivide(divide_ch, divide_ch,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends, stride=2)
        self.att4 = AttentionBlockDivide(divide_ch, divide_ch_x0_5,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends,
                                        shared_ch_start_x0_5, shared_ch_end_x0_5, specific_ch_starts_x0_5, specific_ch_ends_x0_5)
        self.res5 = ResidualBlockDivide(divide_ch, divide_ch,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends)
        self.att6 = AttentionBlockDivide(divide_ch, divide_ch_x0_5,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends,
                                        shared_ch_start_x0_5, shared_ch_end_x0_5, specific_ch_starts_x0_5, specific_ch_ends_x0_5)
        self.res7 = ResidualBlockDivide(divide_ch, divide_ch,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends)
        self.res8 = ResidualBlockUpsampleDivide(divide_ch, divide_ch,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends, 2)
        self.res9 = ResidualBlockDivide(divide_ch, divide_ch,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends)
        self.conv10 = subpel_conv_divide(divide_ch, io_ch,
                                        shared_ch_start, shared_ch_end, specific_ch_starts, specific_ch_ends,
                                        io_shared_ch_start, io_shared_ch_end, io_specific_ch_starts, io_specific_ch_ends, upscale_factor=2, kernel_size=3, padding=1)

    def forward(self, x, index_channel=0):
        out = x
        out = self.res1(out, index_channel=index_channel)
        out = self.res2(out, index_channel=index_channel)
        out = self.res3(out, index_channel=index_channel)
        out = self.att4(out, index_channel=index_channel)
        out = self.res5(out, index_channel=index_channel)
        out = self.att6(out, index_channel=index_channel)
        out = self.res7(out, index_channel=index_channel)
        out = self.res8(out, index_channel=index_channel)
        out = self.res9(out, index_channel=index_channel)
        out = self.conv10(out, index_channel=index_channel)
        out = x + out

        return out


def channel_mult(input, coeff):
    output = []
    for index in range(len(input)):
        output.append(math.floor(input[index] * coeff))

    return output

