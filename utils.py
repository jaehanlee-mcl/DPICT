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

import argparse
import math
import random
import shutil
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ms_ssim

from compressai.datasets import ImageFolder
from compressai.zoo import models

class minimum_resize(object):
    def __init__(self, patch_size):
        self.patch_height = patch_size[0]
        self.patch_width = patch_size[1]

    def __call__(self, image):

        width = image.size[0]
        height = image.size[1]
        if height < self.patch_height:
            height_ratio = self.patch_height / height
        else:
            height_ratio = 1

        if width < self.patch_width:
            width_ratio = self.patch_width / width
        else:
            width_ratio = 1

        ratio = max(height_ratio, width_ratio)
        if ratio > 1:
            width = math.ceil(ratio * width)
            height = math.ceil(ratio * height)
            image = image.resize(size=[width, height])

        return image

class random_resize(object):
    def __init__(self, patch_size):
        self.patch_height = patch_size[0]
        self.patch_width = patch_size[1]

    def __call__(self, image):

        width = image.size[0]
        height = image.size[1]

        height_max_range = math.log2(height / self.patch_height)
        width_max_range = math.log2(width / self.patch_width)

        height_select = random.uniform(0, height_max_range)
        width_select = random.uniform(0, width_max_range)

        width = math.ceil(self.patch_width * pow(2, width_select))
        height = math.ceil(self.patch_height * pow(2, height_select))
        image = image.resize(size=[width, height])

        return image

def get_LSB_alpha(epoch, LSB_start, LSB_slope, LSB_min, LSB_max):

    if epoch == 0:
        LSB_alpha = LSB_start
    if epoch > 0:
        LSB_alpha = LSB_start + LSB_slope * (epoch-1)

    LSB_alpha = min(LSB_alpha, LSB_max)
    LSB_alpha = max(LSB_alpha, LSB_min)

    return LSB_alpha

class WeightsRebalancer():
    def __init__(self):
        super().__init__()
        self.weights_prev = None
        self.weights_current = None
        self.losses_prev = None
        self.losses_current = None
        self.overall_loss_prev = None
        self.overall_loss_current = None

    def update(self, weights, losses, overall_loss):

        self.weights_prev = self.weights_current
        self.losses_prev = self.losses_current
        self.overall_loss_prev = self.overall_loss_current

        self.weights_current = weights
        self.losses_current = losses
        self.overall_loss_current = overall_loss

    def rebalance(self, initial_scales, alpha=1):

        if self.weights_prev is None:
            overall_loss_prev = np.mean(self.overall_loss_current, axis=0)
            overall_loss_current = np.mean(self.overall_loss_current, axis=0)
            losses_prev = np.mean(self.losses_current, axis=0)
            losses_current = np.mean(self.losses_current, axis=0)
        elif self.weights_prev is not None:
            overall_loss_prev = np.mean(self.overall_loss_prev, axis=0)
            overall_loss_current = np.mean(self.overall_loss_current, axis=0)
            losses_prev = np.mean(self.losses_prev, axis=0)
            losses_current = np.mean(self.losses_current, axis=0)

        portion_prev = losses_prev / overall_loss_prev
        portion_current = losses_current / overall_loss_current

        difficulty = portion_current / portion_prev

        weights_alpha0 = initial_scales / portion_current
        magnitude_alpha0 = np.sum(weights_alpha0 * losses_current)

        weights_with_difficuty = weights_alpha0 * np.power(difficulty, alpha)
        magnitude_with_difficulty = np.sum(weights_with_difficuty * losses_current)
        magnitude_ratio = magnitude_alpha0 / magnitude_with_difficulty

        weights = weights_with_difficuty * magnitude_ratio

        print("  ")
        print("alpha:  ", alpha)
        print("  ")
        print("portion(prev):  ", portion_prev)
        print("portion(current):  ", portion_current)
        print("difficulty:  ", difficulty)
        print("  ")
        print("weights(current):  ", self.weights_current)
        print("weights(alpha0):  ", weights_alpha0)
        print("weights(next):  ", weights)
        print("magnitude ratio:  ", magnitude_ratio)
        print("  ")
        print("loss(current):  ", np.sum(self.weights_current * losses_current))
        print("loss(alpha0):  ", np.sum(weights_alpha0 * losses_current))
        print("loss(next):  ", np.sum(weights * losses_current))
        print("  ")

        return weights



class LossRecoder():
    def __init__(self, losses):
        super().__init__()
        self.losses = losses

    def update_losses(self, losses):
        self.losses = np.concatenate((self.losses, losses), axis=0)

    def update_overall_loss(self, loss_weights):
        self.overall_loss = np.sum(self.losses * np.expand_dims(loss_weights, axis=0), axis=1)


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, use_ms_ssim=False):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        if use_ms_ssim == False:
            self.weight_mse = 1
            self.weight_ms_ssim = 0
        elif use_ms_ssim == True:
            self.weight_mse = 0
            self.weight_ms_ssim = 1

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = H * W

        index = 0
        for likelihoods in output["likelihoods"].values():
            if index == 0:
                out["bpp_loss"] = torch.sum(torch.log(likelihoods), dim=[1,2,3]) / (-math.log(2) * num_pixels)
                index += 1
            elif index > 0:
                out["bpp_loss"] = out["bpp_loss"] + torch.sum(torch.log(likelihoods), dim=[1,2,3]) / (-math.log(2) * num_pixels)
                index += 1

        out["mse_loss"] = (255 ** 2) * torch.mean(torch.pow(output["x_hat"] - target, 2), dim=[1,2,3])
        out["PSNR"] = -10 * np.log10(out["mse_loss"].cpu().detach().numpy() / (255 ** 2))

        out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0, size_average=False)
        out["MS-SSIM"] = 1 - out["ms_ssim_loss"].cpu().detach().numpy()
        out["MS-SSIM-DB"] = -10 * np.log10(1 - out["MS-SSIM"])

        out["loss"] \
            = self.lmbda * out["mse_loss"] * self.weight_mse \
              + self.lmbda * out["ms_ssim_loss"] * self.weight_ms_ssim \
              + out["bpp_loss"]

        return out


class DistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}

        out["mse_loss"] = (255 ** 2) * torch.mean(torch.pow(output["x_hat"] - target, 2), dim=[1,2,3])
        out["PSNR"] = -10 * np.log10(out["mse_loss"].cpu().detach().numpy() / (255 ** 2))

        out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0, size_average=False)
        out["MS-SSIM"] = 1 - out["ms_ssim_loss"].cpu().detach().numpy()
        out["MS-SSIM-DB"] = -10 * np.log10(1 - out["MS-SSIM"])

        out["loss"] = out["mse_loss"] * 1.0 + out["ms_ssim_loss"] * 0.0

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

def train_DPICT_main(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, base_lr, clip_max_norm, quantize_parameters, quantize_randomize_parameters, loss_weights, distillation=True,
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        if i == 100:
            a=0
            a=0+1
            a=a+3

        epoch_progress_rate = i / len(train_dataloader)
        lr_multiplier = math.cos(epoch_progress_rate * math.pi) / 2 + 0.5
        lr = base_lr * lr_multiplier
        optimizer.param_groups[0]['lr'] = lr

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_criterion = []
        for index_divide in range(len(criterion)):
            if index_divide == 0:
                d_for_loss = d.detach()
            elif index_divide > 0 and distillation == True:
                d_for_loss = out_net['x_hat'].detach()
            elif index_divide > 0 and distillation == False:
                d_for_loss = d.detach()

            current_quantize_parameters \
                = quantize_parameters[index_divide] \
                  + np.random.randn() * quantize_randomize_parameters[index_divide]
            out_net = model(d, index_channel=index_divide, quantize_parameters=[current_quantize_parameters, 0, 0, 0])

            out_criterion.append(criterion[index_divide](out_net, d_for_loss))
            loss_this_index = out_criterion[index_divide]["loss"]
            torch.sum(loss_weights[index_divide] * loss_this_index).backward()

            if index_divide == 0:
                loss_across_index = np.expand_dims(loss_this_index.cpu().detach().numpy(), axis=1)
            elif index_divide > 0:
                loss_across_index = np.concatenate((loss_across_index, np.expand_dims(loss_this_index.cpu().detach().numpy(), axis=1)), axis=1)

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i == 0:
            loss_recoder = LossRecoder(loss_across_index)
        elif i > 0:
            loss_recoder.update_losses(loss_across_index)

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f"\tLearning rate: {optimizer.param_groups[0]['lr']:.7f} |"
                f"\tAux loss: {aux_loss.item():.2f} |"
            )
            for index_divide in range(len(criterion)):
                print(
                    f'\tLoss: {torch.mean(out_criterion[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss: {torch.mean(out_criterion[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss: {torch.mean(out_criterion[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f'\tBpp loss: {torch.mean(out_criterion[index_divide]["bpp_loss"]).item():.5f} |'
                    f"\tPSNR: {np.mean(out_criterion[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM: {np.mean(out_criterion[index_divide]['MS-SSIM']).item():.4f} |"
                    f"\tMS-SSIM (DB): {np.mean(out_criterion[index_divide]['MS-SSIM-DB']).item():.2f}"
                )

    loss_recoder.update_overall_loss(loss_weights)
    optimizer.param_groups[0]['lr'] = base_lr

    return loss_recoder.losses, loss_recoder.overall_loss

def train_DPICT_post(
        model_addon_rear_q2,
        model_addon_rear_q3,
        criterion,
        train_dataloader,
        optimizer_addon_rear_q2,
        optimizer_addon_rear_q3,
        aux_optimizer_addon_rear_q2,
        aux_optimizer_addon_rear_q3,
        epoch, base_lr, clip_max_norm,
        loss_weights,
):
    model_addon_rear_q2.train()
    model_addon_rear_q3.train()

    device = next(model_addon_rear_q2.parameters()).device

    for i, d in enumerate(train_dataloader):
        if i == 100:
            a=0

        epoch_progress_rate = i / len(train_dataloader)
        lr_multiplier = math.cos(epoch_progress_rate * math.pi) / 2 + 0.5
        lr = base_lr * lr_multiplier
        optimizer_addon_rear_q2.param_groups[0]['lr'] = lr
        optimizer_addon_rear_q3.param_groups[0]['lr'] = lr

        optimizer_addon_rear_q2.zero_grad()
        optimizer_addon_rear_q3.zero_grad()

        out_criterion_rear_q2 = []
        out_criterion_rear_q3 = []

        out_criterion_base_q2 = []
        out_criterion_base_q3 = []

        batch_size = d['original'].shape[0]
        d_for_original = d['original'].to(device).detach()
        d_for_q0 = d['q0'].to(device).detach()
        d_for_q2 = d['q2'].to(device).detach()
        d_for_q3 = d['q3'].to(device).detach()

        index_divide = 0

        ## BASE performance
        out_net_base_q2 = {"x_hat": d_for_q2}
        out_net_base_q3 = {"x_hat": d_for_q3}
        out_criterion_base_q2.append(criterion[index_divide](out_net_base_q2, d_for_q0))
        out_criterion_base_q3.append(criterion[index_divide](out_net_base_q3, d_for_q0))

        # q2
        out_net_addon_rear_q2 = model_addon_rear_q2(d_for_q2, index_channel=index_divide)
        out_criterion_rear_q2.append(criterion[index_divide](out_net_addon_rear_q2, d_for_q0))
        loss_this_index_rear_q2 = out_criterion_rear_q2[index_divide]["loss"]
        torch.sum(loss_weights[index_divide] * loss_this_index_rear_q2).backward()
        loss_across_index_rear_q2 = np.expand_dims(loss_this_index_rear_q2.cpu().detach().numpy(), axis=1)
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_addon_rear_q2.parameters(), clip_max_norm)
        optimizer_addon_rear_q2.step()
        aux_loss_rear_q2 = model_addon_rear_q2.aux_loss()
        aux_loss_rear_q2.backward()
        aux_optimizer_addon_rear_q2.step()

        # q3
        out_net_addon_rear_q3 = model_addon_rear_q3(d_for_q3, index_channel=index_divide)
        out_criterion_rear_q3.append(criterion[index_divide](out_net_addon_rear_q3, d_for_q0))
        loss_this_index_rear_q3 = out_criterion_rear_q3[index_divide]["loss"]
        torch.sum(loss_weights[index_divide] * loss_this_index_rear_q3).backward()
        loss_across_index_rear_q3 = np.expand_dims(loss_this_index_rear_q3.cpu().detach().numpy(), axis=1)
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_addon_rear_q3.parameters(), clip_max_norm)
        optimizer_addon_rear_q3.step()
        aux_loss_rear_q3 = model_addon_rear_q3.aux_loss()
        aux_loss_rear_q3.backward()
        aux_optimizer_addon_rear_q3.step()

        if i == 0:
            loss_recoder_rear_q2 = LossRecoder(loss_across_index_rear_q2)
            loss_recoder_rear_q3 = LossRecoder(loss_across_index_rear_q3)
        elif i > 0:
            loss_recoder_rear_q2.update_losses(loss_across_index_rear_q2)
            loss_recoder_rear_q3.update_losses(loss_across_index_rear_q3)

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*batch_size}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] \n"
                f"\tLearning rate Q2: {optimizer_addon_rear_q2.param_groups[0]['lr']:.7f} |"
                f"\tAux loss Rear Q2: {aux_loss_rear_q2.item():.2f} || \n"
                f"\tLearning rate Q3: {optimizer_addon_rear_q3.param_groups[0]['lr']:.7f} |"
                f"\tAux loss Rear Q3: {aux_loss_rear_q3.item():.2f} || \n"
            )
            for index_divide in range(len(criterion)):
                print(
                    f'\tQ2 Loss (Rear): {torch.mean(out_criterion_rear_q2[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss : {torch.mean(out_criterion_rear_q2[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss : {torch.mean(out_criterion_rear_q2[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f"\tPSNR : {np.mean(out_criterion_rear_q2[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM (DB) : {np.mean(out_criterion_rear_q2[index_divide]['MS-SSIM-DB']).item():.2f} "
                )
                print(
                    f'\t   Loss (Base): {torch.mean(out_criterion_base_q2[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss : {torch.mean(out_criterion_base_q2[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss : {torch.mean(out_criterion_base_q2[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f"\tPSNR : {np.mean(out_criterion_base_q2[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM (DB) : {np.mean(out_criterion_base_q2[index_divide]['MS-SSIM-DB']).item():.2f} "
                )

                print(
                    f'\tQ3 Loss (Rear): {torch.mean(out_criterion_rear_q3[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss : {torch.mean(out_criterion_rear_q3[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss : {torch.mean(out_criterion_rear_q3[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f"\tPSNR : {np.mean(out_criterion_rear_q3[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM (DB) : {np.mean(out_criterion_rear_q3[index_divide]['MS-SSIM-DB']).item():.2f} "
                )
                print(
                    f'\t   Loss (Base): {torch.mean(out_criterion_base_q3[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss : {torch.mean(out_criterion_base_q3[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss : {torch.mean(out_criterion_base_q3[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f"\tPSNR : {np.mean(out_criterion_base_q3[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM (DB) : {np.mean(out_criterion_base_q3[index_divide]['MS-SSIM-DB']).item():.2f} "
                )

                print("\n")

    loss_recoder_rear_q2.update_overall_loss(loss_weights)
    loss_recoder_rear_q3.update_overall_loss(loss_weights)

    optimizer_addon_rear_q2.param_groups[0]['lr'] = base_lr
    optimizer_addon_rear_q3.param_groups[0]['lr'] = base_lr

    return loss_recoder_rear_q2.losses, loss_recoder_rear_q2.losses, \
           loss_recoder_rear_q3.losses, loss_recoder_rear_q3.losses

def test_DPICT_main(epoch, test_dataloader, model, criterion, quantize_parameters=0, loss_weights=np.array([1/3,1/3,1/3]), distillation=True):
    model.eval()
    device = next(model.parameters()).device

    loss = []
    bpp_loss = []
    mse_loss = []
    msssim_loss = []
    aux_loss = []
    psnr = []
    msssim = []
    msssim_db = []
    for index_divide in range(len(criterion)):
        loss.append(AverageMeter())
        bpp_loss.append(AverageMeter())
        mse_loss.append(AverageMeter())
        msssim_loss.append(AverageMeter())
        aux_loss.append(AverageMeter())
        psnr.append(AverageMeter())
        msssim.append(AverageMeter())
        msssim_db.append(AverageMeter())

    with torch.no_grad():
        i = -1
        for d in test_dataloader:
            i += 1
            d = d.to(device)
            for index_divide in range(len(criterion)):
                if index_divide == 0:
                    d_for_loss = d.detach()
                elif index_divide > 0 and distillation == True:
                    d_for_loss = out_net['x_hat'].detach()
                elif index_divide > 0 and distillation == False:
                    d_for_loss = d.detach()
                out_net = model(d, index_channel=index_divide, quantize_parameters=[quantize_parameters[index_divide], 0, 0, 0])
                out_criterion = criterion[index_divide](out_net, d_for_loss)

                loss_this_index = out_criterion["loss"]
                if index_divide == 0:
                    loss_across_index = np.expand_dims(loss_this_index.cpu().detach().numpy(), axis=1)
                elif index_divide > 0:
                    loss_across_index = np.concatenate((loss_across_index, np.expand_dims(loss_this_index.cpu().detach().numpy(), axis=1)), axis=1)

                aux_loss[index_divide].update(model.aux_loss())
                bpp_loss[index_divide].update(out_criterion["bpp_loss"].mean())
                mse_loss[index_divide].update(out_criterion["mse_loss"].mean())
                msssim_loss[index_divide].update(out_criterion["ms_ssim_loss"].mean())
                loss[index_divide].update(out_criterion["loss"].mean())
                psnr[index_divide].update(out_criterion['PSNR'].mean())
                msssim[index_divide].update(out_criterion['MS-SSIM'].mean())
                msssim_db[index_divide].update(out_criterion['MS-SSIM-DB'].mean())

            if i == 0:
                loss_recoder = LossRecoder(loss_across_index)
            elif i > 0:
                loss_recoder.update_losses(loss_across_index)

    for index_divide in range(len(criterion)):
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss[index_divide].avg:.3f} |"
            f"\tMSE loss: {mse_loss[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss: {msssim_loss[index_divide].avg:.4f} |'
            f"\tBpp loss: {bpp_loss[index_divide].avg:.2f} |"
            f"\tAux loss: {aux_loss[index_divide].avg:.2f} |"
            f"\tPSNR: {psnr[index_divide].avg:.2f} |"
            f"\tMS-SSIM: {msssim[index_divide].avg:.4f} |"
            f"\tMS-SSIM(DB): {msssim_db[index_divide].avg:.2f}\n"
        )

    loss_recoder.update_overall_loss(loss_weights)

    return loss_recoder.losses, loss_recoder.overall_loss

def test_DPICT_post(epoch,
                        test_dataloader,
                        model_addon_rear_q2,
                        model_addon_rear_q3,
                        criterion,
                        loss_weights=np.array([1/3,1/3,1/3]),
                        distillation=True):
    model_addon_rear_q2.eval()
    model_addon_rear_q3.eval()

    device = next(model_addon_rear_q2.parameters()).device

    loss_rear_q2 = []
    mse_loss_rear_q2 = []
    msssim_loss_rear_q2 = []
    psnr_rear_q2 = []
    msssim_db_rear_q2 = []

    loss_rear_q3 = []
    mse_loss_rear_q3 = []
    msssim_loss_rear_q3 = []
    psnr_rear_q3 = []
    msssim_db_rear_q3 = []

    loss_base_q2 = []
    mse_loss_base_q2 = []
    msssim_loss_base_q2 = []
    psnr_base_q2 = []
    msssim_db_base_q2 = []

    loss_base_q3 = []
    mse_loss_base_q3 = []
    msssim_loss_base_q3 = []
    psnr_base_q3 = []
    msssim_db_base_q3 = []

    for index_divide in range(len(criterion)):

        loss_rear_q2.append(AverageMeter())
        mse_loss_rear_q2.append(AverageMeter())
        msssim_loss_rear_q2.append(AverageMeter())
        psnr_rear_q2.append(AverageMeter())
        msssim_db_rear_q2.append(AverageMeter())

        loss_rear_q3.append(AverageMeter())
        mse_loss_rear_q3.append(AverageMeter())
        msssim_loss_rear_q3.append(AverageMeter())
        psnr_rear_q3.append(AverageMeter())
        msssim_db_rear_q3.append(AverageMeter())

        loss_base_q2.append(AverageMeter())
        mse_loss_base_q2.append(AverageMeter())
        msssim_loss_base_q2.append(AverageMeter())
        psnr_base_q2.append(AverageMeter())
        msssim_db_base_q2.append(AverageMeter())

        loss_base_q3.append(AverageMeter())
        mse_loss_base_q3.append(AverageMeter())
        msssim_loss_base_q3.append(AverageMeter())
        psnr_base_q3.append(AverageMeter())
        msssim_db_base_q3.append(AverageMeter())

    with torch.no_grad():
        i = -1
        for d in test_dataloader:
            i += 1
            batch_size = d['original'].shape[0]
            d_for_original = d['original'].to(device).detach()
            d_for_q0 = d['q0'].to(device).detach()
            d_for_q2 = d['q2'].to(device).detach()
            d_for_q3 = d['q3'].to(device).detach()

            if distillation == True:
                d_for_loss = d_for_q0
            elif distillation == False:
                d_for_loss = d_for_original

            index_divide = 0

            # REAR performance
            out_net_addon_rear_q2 = model_addon_rear_q2(d_for_q2, index_channel=index_divide)
            out_net_addon_rear_q3 = model_addon_rear_q3(d_for_q3, index_channel=index_divide)

            out_criterion_rear_q2 = criterion[index_divide](out_net_addon_rear_q2, d_for_loss)
            out_criterion_rear_q3 = criterion[index_divide](out_net_addon_rear_q3, d_for_loss)

            loss_this_index_rear_q2 = out_criterion_rear_q2["loss"]
            loss_across_index_rear_q2 = np.expand_dims(loss_this_index_rear_q2.cpu().detach().numpy(), axis=1)
            loss_this_index_rear_q3 = out_criterion_rear_q3["loss"]
            loss_across_index_rear_q3 = np.expand_dims(loss_this_index_rear_q3.cpu().detach().numpy(), axis=1)

            loss_rear_q2[index_divide].update(out_criterion_rear_q2["loss"].mean())
            mse_loss_rear_q2[index_divide].update(out_criterion_rear_q2["mse_loss"].mean())
            msssim_loss_rear_q2[index_divide].update(out_criterion_rear_q2["ms_ssim_loss"].mean())
            psnr_rear_q2[index_divide].update(out_criterion_rear_q2['PSNR'].mean())
            msssim_db_rear_q2[index_divide].update(out_criterion_rear_q2['MS-SSIM-DB'].mean())

            loss_rear_q3[index_divide].update(out_criterion_rear_q3["loss"].mean())
            mse_loss_rear_q3[index_divide].update(out_criterion_rear_q3["mse_loss"].mean())
            msssim_loss_rear_q3[index_divide].update(out_criterion_rear_q3["ms_ssim_loss"].mean())
            psnr_rear_q3[index_divide].update(out_criterion_rear_q3['PSNR'].mean())
            msssim_db_rear_q3[index_divide].update(out_criterion_rear_q3['MS-SSIM-DB'].mean())

            if i == 0:
                loss_recoder_rear_q2 = LossRecoder(loss_across_index_rear_q2)
                loss_recoder_rear_q3 = LossRecoder(loss_across_index_rear_q3)
            elif i > 0:
                loss_recoder_rear_q2.update_losses(loss_across_index_rear_q2)
                loss_recoder_rear_q3.update_losses(loss_across_index_rear_q3)


            ## BASE performance
            out_net_base_q2 = {"x_hat": d_for_q2}
            out_net_base_q3 = {"x_hat": d_for_q3}

            out_criterion_base_q2 = criterion[index_divide](out_net_base_q2, d_for_loss)
            out_criterion_base_q3 = criterion[index_divide](out_net_base_q3, d_for_loss)

            loss_this_index_base_q2 = out_criterion_base_q2["loss"]
            loss_across_index_base_q2 = np.expand_dims(loss_this_index_base_q2.cpu().detach().numpy(), axis=1)
            loss_this_index_base_q3 = out_criterion_base_q3["loss"]
            loss_across_index_base_q3 = np.expand_dims(loss_this_index_base_q3.cpu().detach().numpy(), axis=1)

            loss_base_q2[index_divide].update(out_criterion_base_q2["loss"].mean())
            mse_loss_base_q2[index_divide].update(out_criterion_base_q2["mse_loss"].mean())
            msssim_loss_base_q2[index_divide].update(out_criterion_base_q2["ms_ssim_loss"].mean())
            psnr_base_q2[index_divide].update(out_criterion_base_q2['PSNR'].mean())
            msssim_db_base_q2[index_divide].update(out_criterion_base_q2['MS-SSIM-DB'].mean())

            loss_base_q3[index_divide].update(out_criterion_base_q3["loss"].mean())
            mse_loss_base_q3[index_divide].update(out_criterion_base_q3["mse_loss"].mean())
            msssim_loss_base_q3[index_divide].update(out_criterion_base_q3["ms_ssim_loss"].mean())
            psnr_base_q3[index_divide].update(out_criterion_base_q3['PSNR'].mean())
            msssim_db_base_q3[index_divide].update(out_criterion_base_q3['MS-SSIM-DB'].mean())

            if i == 0:
                loss_recoder_base_q2 = LossRecoder(loss_across_index_base_q2)
                loss_recoder_base_q3 = LossRecoder(loss_across_index_base_q3)
            elif i > 0:
                loss_recoder_base_q2.update_losses(loss_across_index_base_q2)
                loss_recoder_base_q3.update_losses(loss_across_index_base_q3)


    for index_divide in range(len(criterion)):
        print(
            f"Test epoch {epoch}: Average losses: "
        )

        print(
            f"\tQ2 Loss (Rear): {loss_rear_q2[index_divide].avg:.3f} |"
            f"\tMSE loss: {mse_loss_rear_q2[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss: {msssim_loss_rear_q2[index_divide].avg:.4f} |'
            f"\tPSNR: {psnr_rear_q2[index_divide].avg:.2f} |"
            f"\tMS-SSIM(DB): {msssim_db_rear_q2[index_divide].avg:.2f} "
        )
        print(
            f"\t   Loss (Base): {loss_base_q2[index_divide].avg:.3f} |"
            f"\tMSE loss: {mse_loss_base_q2[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss: {msssim_loss_base_q2[index_divide].avg:.4f} |'
            f"\tPSNR: {psnr_base_q2[index_divide].avg:.2f} |"
            f"\tMS-SSIM(DB): {msssim_db_base_q2[index_divide].avg:.2f} "
        )

        print(
            f"\tQ3 Loss (Rear): {loss_rear_q3[index_divide].avg:.3f} |"
            f"\tMSE loss: {mse_loss_rear_q3[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss: {msssim_loss_rear_q3[index_divide].avg:.4f} |'
            f"\tPSNR: {psnr_rear_q3[index_divide].avg:.2f} |"
            f"\tMS-SSIM(DB): {msssim_db_rear_q3[index_divide].avg:.2f} "
        )
        print(
            f"\t   Loss (Base): {loss_base_q3[index_divide].avg:.3f} |"
            f"\tMSE loss: {mse_loss_base_q3[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss: {msssim_loss_base_q3[index_divide].avg:.4f} |'
            f"\tPSNR: {psnr_base_q3[index_divide].avg:.2f} |"
            f"\tMS-SSIM(DB): {msssim_db_base_q3[index_divide].avg:.2f} "
        )

    loss_recoder_rear_q2.update_overall_loss(loss_weights)
    loss_recoder_rear_q3.update_overall_loss(loss_weights)

    return loss_recoder_rear_q2.losses, loss_recoder_rear_q2.overall_loss,\
           loss_recoder_rear_q3.losses, loss_recoder_rear_q3.overall_loss

def save_checkpoint(state, is_best, filedir="", filename="checkpoint", suffix=""):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filedir + '/best_loss' + suffix + '.pth.tar')