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


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tMS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f} |"
                f"\tPSNR: {out_criterion['PSNR']:.2f} |"
                f"\tMS-SSIM: {out_criterion['MS-SSIM']:.4f} |"
                f"\tMS-SSIM (DB): {out_criterion['MS-SSIM-DB']:.2f}"
            )


def train_slim_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_criterion = []
        for index_slim in range(len(criterion)):
            out_net = model(d, index_channel=index_slim)

            out_criterion.append(criterion[index_slim](out_net, d))
            out_criterion[index_slim]["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f"\tAux loss: {aux_loss.item():.2f} |"
            )
            for index_slim in range(len(criterion)):
                print(
                    f'\tLoss: {out_criterion[index_slim]["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion[index_slim]["mse_loss"].item():.3f} |'
                    f'\tMS-SSIM loss: {out_criterion[index_slim]["ms_ssim_loss"].item():.4f} |'
                    f'\tBpp loss: {out_criterion[index_slim]["bpp_loss"].item():.5f} |'
                    f"\tPSNR: {out_criterion[index_slim]['PSNR']:.2f} |"
                    f"\tMS-SSIM: {out_criterion[index_slim]['MS-SSIM']:.4f} |"
                    f"\tMS-SSIM (DB): {out_criterion[index_slim]['MS-SSIM-DB']:.2f}"
                )


def train_slim_one_epoch_with_quantize_parameters(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, quantize_parameters
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_criterion = []
        for index_slim in range(len(criterion)):
            out_net = model(d, index_channel=index_slim, quantize_parameters=[quantize_parameters[index_slim], 0, 0, 0])

            out_criterion.append(criterion[index_slim](out_net, d))
            out_criterion[index_slim]["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f"\tAux loss: {aux_loss.item():.2f} |"
            )
            for index_slim in range(len(criterion)):
                print(
                    f'\tLoss: {out_criterion[index_slim]["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion[index_slim]["mse_loss"].item():.3f} |'
                    f'\tMS-SSIM loss: {out_criterion[index_slim]["ms_ssim_loss"].item():.4f} |'
                    f'\tBpp loss: {out_criterion[index_slim]["bpp_loss"].item():.5f} |'
                    f"\tPSNR: {out_criterion[index_slim]['PSNR']:.2f} |"
                    f"\tMS-SSIM: {out_criterion[index_slim]['MS-SSIM']:.4f} |"
                    f"\tMS-SSIM (DB): {out_criterion[index_slim]['MS-SSIM-DB']:.2f}"
                )


def train_slim_one_epoch_with_quantize_randomize_parameters(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, quantize_parameters, quantize_randomize_parameters,
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_criterion = []
        for index_slim in range(len(criterion)):
            current_quantize_parameters \
                = quantize_parameters[index_slim] \
                  + np.random.uniform(
                  low=-quantize_randomize_parameters[index_slim]/2,
                  high=quantize_randomize_parameters[index_slim]/2,)
            out_net = model(d, index_channel=index_slim, quantize_parameters=[current_quantize_parameters, 0, 0, 0])

            out_criterion.append(criterion[index_slim](out_net, d))
            out_criterion[index_slim]["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f"\tAux loss: {aux_loss.item():.2f} |"
            )
            for index_slim in range(len(criterion)):
                print(
                    f'\tLoss: {out_criterion[index_slim]["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion[index_slim]["mse_loss"].item():.3f} |'
                    f'\tMS-SSIM loss: {out_criterion[index_slim]["ms_ssim_loss"].item():.4f} |'
                    f'\tBpp loss: {out_criterion[index_slim]["bpp_loss"].item():.5f} |'
                    f"\tPSNR: {out_criterion[index_slim]['PSNR']:.2f} |"
                    f"\tMS-SSIM: {out_criterion[index_slim]['MS-SSIM']:.4f} |"
                    f"\tMS-SSIM (DB): {out_criterion[index_slim]['MS-SSIM-DB']:.2f}"
                )


def train_slide_one_epoch_with_quantize_randomize_parameters(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, quantize_parameters, quantize_randomize_parameters,
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_criterion = []
        for index_slide in range(len(criterion)):
            current_quantize_parameters \
                = quantize_parameters[index_slide] \
                  + np.random.randn() * quantize_randomize_parameters[index_slide]
            out_net = model(d, index_channel=index_slide, quantize_parameters=[current_quantize_parameters, 0, 0, 0])

            out_criterion.append(criterion[index_slide](out_net, d))
            out_criterion[index_slide]["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f"\tAux loss: {aux_loss.item():.2f} |"
            )
            for index_slide in range(len(criterion)):
                print(
                    f'\tLoss: {out_criterion[index_slide]["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion[index_slide]["mse_loss"].item():.3f} |'
                    f'\tMS-SSIM loss: {out_criterion[index_slide]["ms_ssim_loss"].item():.4f} |'
                    f'\tBpp loss: {out_criterion[index_slide]["bpp_loss"].item():.5f} |'
                    f"\tPSNR: {out_criterion[index_slide]['PSNR']:.2f} |"
                    f"\tMS-SSIM: {out_criterion[index_slide]['MS-SSIM']:.4f} |"
                    f"\tMS-SSIM (DB): {out_criterion[index_slide]['MS-SSIM-DB']:.2f}"
                )

def train_divide_one_epoch_with_quantize_randomize_parameters(
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

    return loss_recoder.losses, loss_recoder.overall_loss

def train_addon_one_epoch(
    model, model_addon_rear, model_addon_side,
        criterion, train_dataloader,
        optimizer_addon_rear, optimizer_addon_side, aux_optimizer_addon_rear, aux_optimizer_addon_side,
        epoch, base_lr, clip_max_norm,
        quantize_parameters, quantize_randomize_parameters,
        quantize_parameters_addon, quantize_randomize_parameters_addon,
        loss_weights, distillation=True,
):
    model.eval()
    model_addon_rear.train()
    model_addon_side.train()

    device = next(model_addon_rear.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        if i == 100:
            a=0
            a=0+1
            a=a+3

        epoch_progress_rate = i / len(train_dataloader)
        lr_multiplier = math.cos(epoch_progress_rate * math.pi) / 2 + 0.5
        lr = base_lr * lr_multiplier
        optimizer_addon_rear.param_groups[0]['lr'] = lr
        optimizer_addon_side.param_groups[0]['lr'] = lr

        optimizer_addon_rear.zero_grad()
        optimizer_addon_rear.zero_grad()
        aux_optimizer_addon_side.zero_grad()
        aux_optimizer_addon_side.zero_grad()

        reference_criterion = []
        out_criterion_rear = []
        out_criterion_side = []
        for index_divide in range(len(criterion)):
            current_quantize_parameters \
                = quantize_parameters[index_divide] \
                  + np.random.randn() * quantize_randomize_parameters[index_divide]
            current_quantize_parameters_addon \
                = quantize_parameters_addon[index_divide] \
                  + np.random.randn() * quantize_randomize_parameters_addon[index_divide]

            if distillation == True:
                out_net_1 = model(d, index_channel=index_divide, quantize_parameters=[current_quantize_parameters, 0, 0, 0])
                d_for_loss = out_net_1['x_hat'].detach()
            elif distillation == False:
                d_for_loss = d.detach()

            out_net_2 = model(d, index_channel=index_divide, quantize_parameters=[current_quantize_parameters_addon, 0, 0, 0], get_y_hat=True)
            d_for_addon_rear = out_net_2['x_hat'].detach()
            d_for_addon_side = out_net_2['y_hat'].detach()

            out_net_addon_rear = model_addon_rear(d_for_addon_rear, index_channel=index_divide, quantize_parameters=[current_quantize_parameters, 0, 0, 0])
            out_net_addon_side = model_addon_side(d_for_addon_side, index_channel=index_divide, quantize_parameters=[current_quantize_parameters, 0, 0, 0])

            reference_criterion.append(criterion[index_divide](out_net_2, d_for_loss))
            out_criterion_rear.append(criterion[index_divide](out_net_addon_rear, d_for_loss))
            out_criterion_side.append(criterion[index_divide](out_net_addon_side, d_for_loss))

            loss_this_index_rear = out_criterion_rear[index_divide]["loss"]
            torch.sum(loss_weights[index_divide] * loss_this_index_rear).backward()
            loss_this_index_side = out_criterion_side[index_divide]["loss"]
            torch.sum(loss_weights[index_divide] * loss_this_index_side).backward()

            if index_divide == 0:
                loss_across_index_rear = np.expand_dims(loss_this_index_rear.cpu().detach().numpy(), axis=1)
                loss_across_index_side = np.expand_dims(loss_this_index_side.cpu().detach().numpy(), axis=1)
            elif index_divide > 0:
                loss_across_index_rear = np.concatenate((loss_across_index_rear, np.expand_dims(loss_this_index_rear.cpu().detach().numpy(), axis=1)), axis=1)
                loss_across_index_side = np.concatenate((loss_across_index_side, np.expand_dims(loss_this_index_side.cpu().detach().numpy(), axis=1)), axis=1)

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_addon_rear.parameters(), clip_max_norm)
            torch.nn.utils.clip_grad_norm_(model_addon_side.parameters(), clip_max_norm)
        optimizer_addon_rear.step()
        optimizer_addon_side.step()

        aux_loss_rear = model_addon_rear.aux_loss()
        aux_loss_rear.backward()
        aux_optimizer_addon_rear.step()

        aux_loss_side = model_addon_side.aux_loss()
        aux_loss_side.backward()
        aux_optimizer_addon_side.step()

        if i == 0:
            loss_recoder_rear = LossRecoder(loss_across_index_rear)
            loss_recoder_side = LossRecoder(loss_across_index_side)
        elif i > 0:
            loss_recoder_rear.update_losses(loss_across_index_rear)
            loss_recoder_side.update_losses(loss_across_index_side)

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f"\tLearning rate: {optimizer_addon_rear.param_groups[0]['lr']:.7f} |"
                f"\tAux loss Rear: {aux_loss_rear.item():.2f} |"
                f"\tAux loss Side: {aux_loss_side.item():.2f} |"
            )
            for index_divide in range(len(criterion)):
                print(
                    f'\tLoss (Rear): {torch.mean(out_criterion_rear[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss (Rear): {torch.mean(out_criterion_rear[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss (Rear): {torch.mean(out_criterion_rear[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f"\tPSNR (Rear): {np.mean(out_criterion_rear[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM (DB) (Rear): {np.mean(out_criterion_rear[index_divide]['MS-SSIM-DB']).item():.2f} \n"
                    
                    f'\tLoss (Side): {torch.mean(out_criterion_side[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss (Side): {torch.mean(out_criterion_side[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss (Side): {torch.mean(out_criterion_side[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f"\tPSNR (Side): {np.mean(out_criterion_side[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM (DB) (Side): {np.mean(out_criterion_side[index_divide]['MS-SSIM-DB']).item():.2f} \n"
                    
                    f'\tLoss (Refe): {torch.mean(reference_criterion[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss (Refe): {torch.mean(reference_criterion[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss (Refe): {torch.mean(reference_criterion[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f"\tPSNR (Refe): {np.mean(reference_criterion[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM (DB) (Refe): {np.mean(reference_criterion[index_divide]['MS-SSIM-DB']).item():.2f}"
                )

    loss_recoder_rear.update_overall_loss(loss_weights)
    loss_recoder_side.update_overall_loss(loss_weights)

    return loss_recoder_rear.losses, loss_recoder_rear.losses, loss_recoder_side.overall_loss, loss_recoder_side.overall_loss

def train_addon_one_epoch_v2(
        model_addon_rear_q1,
        model_addon_rear_q2,
        model_addon_rear_q3,
        model_addon_rear_q4,
        model_addon_rear_q5,
        criterion,
        train_dataloader,
        optimizer_addon_rear_q1,
        optimizer_addon_rear_q2,
        optimizer_addon_rear_q3,
        optimizer_addon_rear_q4,
        optimizer_addon_rear_q5,
        aux_optimizer_addon_rear_q1,
        aux_optimizer_addon_rear_q2,
        aux_optimizer_addon_rear_q3,
        aux_optimizer_addon_rear_q4,
        aux_optimizer_addon_rear_q5,
        epoch, base_lr, clip_max_norm,
        loss_weights,
):
    model_addon_rear_q1.train()
    model_addon_rear_q2.train()
    model_addon_rear_q3.train()
    model_addon_rear_q4.train()
    model_addon_rear_q5.train()

    device = next(model_addon_rear_q1.parameters()).device

    for i, d in enumerate(train_dataloader):
        if i == 100:
            a=0

        epoch_progress_rate = i / len(train_dataloader)
        lr_multiplier = math.cos(epoch_progress_rate * math.pi) / 2 + 0.5
        lr = base_lr * lr_multiplier
        optimizer_addon_rear_q1.param_groups[0]['lr'] = lr
        optimizer_addon_rear_q2.param_groups[0]['lr'] = lr
        optimizer_addon_rear_q3.param_groups[0]['lr'] = lr
        optimizer_addon_rear_q4.param_groups[0]['lr'] = lr
        optimizer_addon_rear_q5.param_groups[0]['lr'] = lr

        optimizer_addon_rear_q1.zero_grad()
        optimizer_addon_rear_q2.zero_grad()
        optimizer_addon_rear_q3.zero_grad()
        optimizer_addon_rear_q4.zero_grad()
        optimizer_addon_rear_q5.zero_grad()

        out_criterion_rear_q1 = []
        out_criterion_rear_q2 = []
        out_criterion_rear_q3 = []
        out_criterion_rear_q4 = []
        out_criterion_rear_q5 = []

        out_criterion_base_q1 = []
        out_criterion_base_q2 = []
        out_criterion_base_q3 = []
        out_criterion_base_q4 = []
        out_criterion_base_q5 = []

        batch_size = d['original'].shape[0]
        d_for_original = d['original'].to(device).detach()
        d_for_q0 = d['q0'].to(device).detach()
        d_for_q1 = d['q1'].to(device).detach()
        d_for_q2 = d['q2'].to(device).detach()
        d_for_q3 = d['q3'].to(device).detach()
        d_for_q4 = d['q4'].to(device).detach()
        d_for_q5 = d['q5'].to(device).detach()

        index_divide = 0

        ## BASE performance
        out_net_base_q1 = {"x_hat": d_for_q1}
        out_net_base_q2 = {"x_hat": d_for_q2}
        out_net_base_q3 = {"x_hat": d_for_q3}
        out_net_base_q4 = {"x_hat": d_for_q4}
        out_net_base_q5 = {"x_hat": d_for_q5}
        out_criterion_base_q1.append(criterion[index_divide](out_net_base_q1, d_for_q0))
        out_criterion_base_q2.append(criterion[index_divide](out_net_base_q2, d_for_q0))
        out_criterion_base_q3.append(criterion[index_divide](out_net_base_q3, d_for_q0))
        out_criterion_base_q4.append(criterion[index_divide](out_net_base_q4, d_for_q0))
        out_criterion_base_q5.append(criterion[index_divide](out_net_base_q5, d_for_q0))

        # q1
        out_net_addon_rear_q1 = model_addon_rear_q1(d_for_q1, index_channel=index_divide)
        out_criterion_rear_q1.append(criterion[index_divide](out_net_addon_rear_q1, d_for_q0))
        loss_this_index_rear_q1 = out_criterion_rear_q1[index_divide]["loss"]
        torch.sum(loss_weights[index_divide] * loss_this_index_rear_q1).backward()
        loss_across_index_rear_q1 = np.expand_dims(loss_this_index_rear_q1.cpu().detach().numpy(), axis=1)
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_addon_rear_q1.parameters(), clip_max_norm)
        optimizer_addon_rear_q1.step()
        aux_loss_rear_q1 = model_addon_rear_q1.aux_loss()
        aux_loss_rear_q1.backward()
        aux_optimizer_addon_rear_q1.step()

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

        # q4
        out_net_addon_rear_q4 = model_addon_rear_q4(d_for_q4, index_channel=index_divide)
        out_criterion_rear_q4.append(criterion[index_divide](out_net_addon_rear_q4, d_for_q0))
        loss_this_index_rear_q4 = out_criterion_rear_q4[index_divide]["loss"]
        torch.sum(loss_weights[index_divide] * loss_this_index_rear_q4).backward()
        loss_across_index_rear_q4 = np.expand_dims(loss_this_index_rear_q4.cpu().detach().numpy(), axis=1)
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_addon_rear_q4.parameters(), clip_max_norm)
        optimizer_addon_rear_q4.step()
        aux_loss_rear_q4 = model_addon_rear_q4.aux_loss()
        aux_loss_rear_q4.backward()
        aux_optimizer_addon_rear_q4.step()

        # q5
        out_net_addon_rear_q5 = model_addon_rear_q5(d_for_q5, index_channel=index_divide)
        out_criterion_rear_q5.append(criterion[index_divide](out_net_addon_rear_q5, d_for_q0))
        loss_this_index_rear_q5 = out_criterion_rear_q5[index_divide]["loss"]
        torch.sum(loss_weights[index_divide] * loss_this_index_rear_q5).backward()
        loss_across_index_rear_q5 = np.expand_dims(loss_this_index_rear_q5.cpu().detach().numpy(), axis=1)
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_addon_rear_q5.parameters(), clip_max_norm)
        optimizer_addon_rear_q5.step()
        aux_loss_rear_q5 = model_addon_rear_q5.aux_loss()
        aux_loss_rear_q5.backward()
        aux_optimizer_addon_rear_q5.step()

        if i == 0:
            loss_recoder_rear_q1 = LossRecoder(loss_across_index_rear_q1)
            loss_recoder_rear_q2 = LossRecoder(loss_across_index_rear_q2)
            loss_recoder_rear_q3 = LossRecoder(loss_across_index_rear_q3)
            loss_recoder_rear_q4 = LossRecoder(loss_across_index_rear_q4)
            loss_recoder_rear_q5 = LossRecoder(loss_across_index_rear_q5)
        elif i > 0:
            loss_recoder_rear_q1.update_losses(loss_across_index_rear_q1)
            loss_recoder_rear_q2.update_losses(loss_across_index_rear_q2)
            loss_recoder_rear_q3.update_losses(loss_across_index_rear_q3)
            loss_recoder_rear_q4.update_losses(loss_across_index_rear_q4)
            loss_recoder_rear_q5.update_losses(loss_across_index_rear_q5)

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*batch_size}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] \n"
                f"\tLearning rate Q1: {optimizer_addon_rear_q1.param_groups[0]['lr']:.7f} |"
                f"\tAux loss Rear Q1: {aux_loss_rear_q1.item():.2f} || \n"
                f"\tLearning rate Q2: {optimizer_addon_rear_q2.param_groups[0]['lr']:.7f} |"
                f"\tAux loss Rear Q2: {aux_loss_rear_q2.item():.2f} || \n"
                f"\tLearning rate Q3: {optimizer_addon_rear_q3.param_groups[0]['lr']:.7f} |"
                f"\tAux loss Rear Q3: {aux_loss_rear_q3.item():.2f} || \n"
                f"\tLearning rate Q4: {optimizer_addon_rear_q4.param_groups[0]['lr']:.7f} |"
                f"\tAux loss Rear Q4: {aux_loss_rear_q4.item():.2f} || \n"
                f"\tLearning rate Q5: {optimizer_addon_rear_q5.param_groups[0]['lr']:.7f} |"
                f"\tAux loss Rear Q5: {aux_loss_rear_q5.item():.2f} || \n"
            )
            for index_divide in range(len(criterion)):
                print(
                    f'\tQ1 Loss (Rear): {torch.mean(out_criterion_rear_q1[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss: {torch.mean(out_criterion_rear_q1[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss: {torch.mean(out_criterion_rear_q1[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f"\tPSNR: {np.mean(out_criterion_rear_q1[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM (DB): {np.mean(out_criterion_rear_q1[index_divide]['MS-SSIM-DB']).item():.2f} "
                )
                print(
                    f'\t   Loss (Base): {torch.mean(out_criterion_base_q1[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss : {torch.mean(out_criterion_base_q1[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss : {torch.mean(out_criterion_base_q1[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f"\tPSNR : {np.mean(out_criterion_base_q1[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM (DB) : {np.mean(out_criterion_base_q1[index_divide]['MS-SSIM-DB']).item():.2f} "
                )

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

                print(
                    f'\tQ4 Loss (Rear): {torch.mean(out_criterion_rear_q4[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss : {torch.mean(out_criterion_rear_q4[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss : {torch.mean(out_criterion_rear_q4[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f"\tPSNR : {np.mean(out_criterion_rear_q4[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM (DB) : {np.mean(out_criterion_rear_q4[index_divide]['MS-SSIM-DB']).item():.2f} "
                )
                print(
                    f'\t   Loss (Base): {torch.mean(out_criterion_base_q4[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss : {torch.mean(out_criterion_base_q4[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss : {torch.mean(out_criterion_base_q4[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f"\tPSNR : {np.mean(out_criterion_base_q4[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM (DB) : {np.mean(out_criterion_base_q4[index_divide]['MS-SSIM-DB']).item():.2f} "
                )

                print(
                    f'\tQ5 Loss (Rear): {torch.mean(out_criterion_rear_q5[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss : {torch.mean(out_criterion_rear_q5[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss : {torch.mean(out_criterion_rear_q5[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f"\tPSNR : {np.mean(out_criterion_rear_q5[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM (DB) : {np.mean(out_criterion_rear_q5[index_divide]['MS-SSIM-DB']).item():.2f} "
                )
                print(
                    f'\t   Loss (Base): {torch.mean(out_criterion_base_q5[index_divide]["loss"]).item():.3f} |'
                    f'\tMSE loss : {torch.mean(out_criterion_base_q5[index_divide]["mse_loss"]).item():.3f} |'
                    f'\tMS-SSIM loss : {torch.mean(out_criterion_base_q5[index_divide]["ms_ssim_loss"]).item():.4f} |'
                    f"\tPSNR : {np.mean(out_criterion_base_q5[index_divide]['PSNR']).item():.2f} |"
                    f"\tMS-SSIM (DB) : {np.mean(out_criterion_base_q5[index_divide]['MS-SSIM-DB']).item():.2f} "
                )

                print("\n")

    loss_recoder_rear_q1.update_overall_loss(loss_weights)
    loss_recoder_rear_q2.update_overall_loss(loss_weights)
    loss_recoder_rear_q3.update_overall_loss(loss_weights)
    loss_recoder_rear_q4.update_overall_loss(loss_weights)
    loss_recoder_rear_q5.update_overall_loss(loss_weights)

    return loss_recoder_rear_q1.losses, loss_recoder_rear_q1.losses, \
           loss_recoder_rear_q2.losses, loss_recoder_rear_q2.losses, \
           loss_recoder_rear_q3.losses, loss_recoder_rear_q3.losses, \
           loss_recoder_rear_q4.losses, loss_recoder_rear_q4.losses,\
           loss_recoder_rear_q5.losses, loss_recoder_rear_q5.losses

def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f'\tMS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} |'
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f} |"
        f"\tPSNR: {out_criterion['PSNR']:.2f} |"
        f"\tMS-SSIM: {out_criterion['MS-SSIM']:.4f} |"
        f"\tMS-SSIM(DB): {out_criterion['MS-SSIM-DB']:.2f}\n"
    )

    return loss.avg

def test_slim_epoch(epoch, test_dataloader, model, criterion, index_slim=0):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d, index_channel=index_slim)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f'\tMS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} |'
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f} |"
        f"\tPSNR: {out_criterion['PSNR']:.2f} |"
        f"\tMS-SSIM: {out_criterion['MS-SSIM']:.4f} |"
        f"\tMS-SSIM(DB): {out_criterion['MS-SSIM-DB']:.2f}\n"
    )

    return loss.avg

def test_slim_epoch_with_quantize_parameters(epoch, test_dataloader, model, criterion, index_slim=0, quantize_parameters=0):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d, index_channel=index_slim, quantize_parameters=[quantize_parameters, 0, 0, 0])
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f'\tMS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} |'
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f} |"
        f"\tPSNR: {out_criterion['PSNR']:.2f} |"
        f"\tMS-SSIM: {out_criterion['MS-SSIM']:.4f} |"
        f"\tMS-SSIM(DB): {out_criterion['MS-SSIM-DB']:.2f}\n"
    )

    return loss.avg

def test_slide_epoch_with_quantize_parameters(epoch, test_dataloader, model, criterion, index_slide=0, quantize_parameters=0):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d, index_channel=index_slide, quantize_parameters=[quantize_parameters, 0, 0, 0])
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f'\tMS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} |'
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f} |"
        f"\tPSNR: {out_criterion['PSNR']:.2f} |"
        f"\tMS-SSIM: {out_criterion['MS-SSIM']:.4f} |"
        f"\tMS-SSIM(DB): {out_criterion['MS-SSIM-DB']:.2f}\n"
    )

    return loss.avg

def test_divide_epoch_with_quantize_parameters(epoch, test_dataloader, model, criterion, quantize_parameters=0, loss_weights=np.array([1/3,1/3,1/3]), distillation=True):
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

def test_epoch_addon(epoch, test_dataloader, model, model_addon_rear, model_addon_side, criterion,
                     quantize_parameters=0, quantize_parameters_addon=0,
                     loss_weights=np.array([1/3,1/3,1/3]), distillation=True):
    model.eval()
    model_addon_rear.eval()
    model_addon_side.eval()
    device = next(model_addon_rear.parameters()).device

    loss_rear = []
    mse_loss_rear = []
    msssim_loss_rear = []
    psnr_rear = []
    msssim_db_rear = []

    loss_side = []
    mse_loss_side = []
    msssim_loss_side = []
    psnr_side = []
    msssim_db_side = []

    loss_reference = []
    mse_loss_reference = []
    msssim_loss_reference = []
    psnr_reference = []
    msssim_db_reference = []

    for index_divide in range(len(criterion)):

        loss_rear.append(AverageMeter())
        mse_loss_rear.append(AverageMeter())
        msssim_loss_rear.append(AverageMeter())
        psnr_rear.append(AverageMeter())
        msssim_db_rear.append(AverageMeter())

        loss_side.append(AverageMeter())
        mse_loss_side.append(AverageMeter())
        msssim_loss_side.append(AverageMeter())
        psnr_side.append(AverageMeter())
        msssim_db_side.append(AverageMeter())

        loss_reference.append(AverageMeter())
        mse_loss_reference.append(AverageMeter())
        msssim_loss_reference.append(AverageMeter())
        psnr_reference.append(AverageMeter())
        msssim_db_reference.append(AverageMeter())

    with torch.no_grad():
        i = -1
        for d in test_dataloader:
            i += 1
            d = d.to(device)
            for index_divide in range(len(criterion)):

                if distillation == True:
                    out_net_1 = model(d, index_channel=index_divide, quantize_parameters=[quantize_parameters[index_divide], 0, 0, 0])
                    d_for_loss = out_net_1['x_hat'].detach()
                elif distillation == False:
                    d_for_loss = d.detach()

                out_net_2 = model(d, index_channel=index_divide, quantize_parameters=[quantize_parameters_addon[index_divide], 0, 0, 0], get_y_hat=True)
                d_for_addon_rear = out_net_2['x_hat'].detach()
                d_for_addon_side = out_net_2['y_hat'].detach()
                out_net_addon_rear = model_addon_rear(d_for_addon_rear, index_channel=index_divide, quantize_parameters=[quantize_parameters_addon[index_divide], 0, 0, 0])
                out_net_addon_side = model_addon_side(d_for_addon_side, index_channel=index_divide, quantize_parameters=[quantize_parameters_addon[index_divide], 0, 0, 0])

                reference_criterion = criterion[index_divide](out_net_2, d_for_loss)
                out_criterion_rear = criterion[index_divide](out_net_addon_rear, d_for_loss)
                out_criterion_side = criterion[index_divide](out_net_addon_side, d_for_loss)

                loss_this_index_rear = out_criterion_rear["loss"]
                loss_this_index_side = out_criterion_side["loss"]
                if index_divide == 0:
                    loss_across_index_rear = np.expand_dims(loss_this_index_rear.cpu().detach().numpy(), axis=1)
                    loss_across_index_side = np.expand_dims(loss_this_index_side.cpu().detach().numpy(), axis=1)
                elif index_divide > 0:
                    loss_across_index_rear = np.concatenate((loss_across_index_rear, np.expand_dims(loss_this_index_rear.cpu().detach().numpy(), axis=1)), axis=1)
                    loss_across_index_side = np.concatenate((loss_across_index_side, np.expand_dims(loss_this_index_side.cpu().detach().numpy(), axis=1)), axis=1)

                loss_rear[index_divide].update(out_criterion_rear["loss"].mean())
                mse_loss_rear[index_divide].update(out_criterion_rear["mse_loss"].mean())
                msssim_loss_rear[index_divide].update(out_criterion_rear["ms_ssim_loss"].mean())
                psnr_rear[index_divide].update(out_criterion_rear['PSNR'].mean())
                msssim_db_rear[index_divide].update(out_criterion_rear['MS-SSIM-DB'].mean())

                loss_side[index_divide].update(out_criterion_side["loss"].mean())
                mse_loss_side[index_divide].update(out_criterion_side["mse_loss"].mean())
                msssim_loss_side[index_divide].update(out_criterion_side["ms_ssim_loss"].mean())
                psnr_side[index_divide].update(out_criterion_side['PSNR'].mean())
                msssim_db_side[index_divide].update(out_criterion_side['MS-SSIM-DB'].mean())

                loss_reference[index_divide].update(reference_criterion["loss"].mean())
                mse_loss_reference[index_divide].update(reference_criterion["mse_loss"].mean())
                msssim_loss_reference[index_divide].update(reference_criterion["ms_ssim_loss"].mean())
                psnr_reference[index_divide].update(reference_criterion['PSNR'].mean())
                msssim_db_reference[index_divide].update(reference_criterion['MS-SSIM-DB'].mean())

            if i == 0:
                loss_recoder_rear = LossRecoder(loss_across_index_rear)
                loss_recoder_side = LossRecoder(loss_across_index_side)
            elif i > 0:
                loss_recoder_rear.update_losses(loss_across_index_rear)
                loss_recoder_side.update_losses(loss_across_index_side)

    for index_divide in range(len(criterion)):
        print(
            f"Test epoch {epoch}: Average losses: \n"
            
            f"\tLoss (Rear): {loss_rear[index_divide].avg:.3f} |"
            f"\tMSE loss (Rear): {mse_loss_rear[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss (Rear): {msssim_loss_rear[index_divide].avg:.4f} |'
            f"\tPSNR (Rear): {psnr_rear[index_divide].avg:.2f} |"
            f"\tMS-SSIM(DB) (Rear): {msssim_db_rear[index_divide].avg:.2f} \n"
            
            f"\tLoss (Side): {loss_side[index_divide].avg:.3f} |"
            f"\tMSE loss (Side): {mse_loss_side[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss (Side): {msssim_loss_side[index_divide].avg:.4f} |'
            f"\tPSNR (Side): {psnr_side[index_divide].avg:.2f} |"
            f"\tMS-SSIM(DB) (Side): {msssim_db_side[index_divide].avg:.2f} \n"
            
            f"\tLoss (Refe): {loss_reference[index_divide].avg:.3f} |"
            f"\tMSE loss (Refe): {mse_loss_reference[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss (Refe): {msssim_loss_reference[index_divide].avg:.4f} |'
            f"\tPSNR (Refe): {psnr_reference[index_divide].avg:.2f} |"
            f"\tMS-SSIM(DB) (Refe): {msssim_db_reference[index_divide].avg:.2f}\n"
        )

    loss_recoder_rear.update_overall_loss(loss_weights)
    loss_recoder_side.update_overall_loss(loss_weights)

    return loss_recoder_rear.losses, loss_recoder_side.losses, loss_recoder_rear.overall_loss, loss_recoder_rear.overall_loss

def test_epoch_addon_v2(epoch,
                        test_dataloader,
                        model_addon_rear_q1,
                        model_addon_rear_q2,
                        model_addon_rear_q3,
                        model_addon_rear_q4,
                        model_addon_rear_q5,
                        criterion,
                        loss_weights=np.array([1/3,1/3,1/3]),
                        distillation=True):
    model_addon_rear_q1.eval()
    model_addon_rear_q2.eval()
    model_addon_rear_q3.eval()
    model_addon_rear_q4.eval()
    model_addon_rear_q5.eval()

    device = next(model_addon_rear_q1.parameters()).device

    loss_rear_q1 = []
    mse_loss_rear_q1 = []
    msssim_loss_rear_q1 = []
    psnr_rear_q1 = []
    msssim_db_rear_q1 = []

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

    loss_rear_q4 = []
    mse_loss_rear_q4 = []
    msssim_loss_rear_q4 = []
    psnr_rear_q4 = []
    msssim_db_rear_q4 = []

    loss_rear_q5 = []
    mse_loss_rear_q5 = []
    msssim_loss_rear_q5 = []
    psnr_rear_q5 = []
    msssim_db_rear_q5 = []

    loss_base_q1 = []
    mse_loss_base_q1 = []
    msssim_loss_base_q1 = []
    psnr_base_q1 = []
    msssim_db_base_q1 = []

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

    loss_base_q4 = []
    mse_loss_base_q4 = []
    msssim_loss_base_q4 = []
    psnr_base_q4 = []
    msssim_db_base_q4 = []

    loss_base_q5 = []
    mse_loss_base_q5 = []
    msssim_loss_base_q5 = []
    psnr_base_q5 = []
    msssim_db_base_q5 = []

    for index_divide in range(len(criterion)):

        loss_rear_q1.append(AverageMeter())
        mse_loss_rear_q1.append(AverageMeter())
        msssim_loss_rear_q1.append(AverageMeter())
        psnr_rear_q1.append(AverageMeter())
        msssim_db_rear_q1.append(AverageMeter())

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

        loss_rear_q4.append(AverageMeter())
        mse_loss_rear_q4.append(AverageMeter())
        msssim_loss_rear_q4.append(AverageMeter())
        psnr_rear_q4.append(AverageMeter())
        msssim_db_rear_q4.append(AverageMeter())

        loss_rear_q5.append(AverageMeter())
        mse_loss_rear_q5.append(AverageMeter())
        msssim_loss_rear_q5.append(AverageMeter())
        psnr_rear_q5.append(AverageMeter())
        msssim_db_rear_q5.append(AverageMeter())

        loss_base_q1.append(AverageMeter())
        mse_loss_base_q1.append(AverageMeter())
        msssim_loss_base_q1.append(AverageMeter())
        psnr_base_q1.append(AverageMeter())
        msssim_db_base_q1.append(AverageMeter())

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

        loss_base_q4.append(AverageMeter())
        mse_loss_base_q4.append(AverageMeter())
        msssim_loss_base_q4.append(AverageMeter())
        psnr_base_q4.append(AverageMeter())
        msssim_db_base_q4.append(AverageMeter())

        loss_base_q5.append(AverageMeter())
        mse_loss_base_q5.append(AverageMeter())
        msssim_loss_base_q5.append(AverageMeter())
        psnr_base_q5.append(AverageMeter())
        msssim_db_base_q5.append(AverageMeter())

    with torch.no_grad():
        i = -1
        for d in test_dataloader:
            i += 1
            batch_size = d['original'].shape[0]
            d_for_original = d['original'].to(device).detach()
            d_for_q0 = d['q0'].to(device).detach()
            d_for_q1 = d['q1'].to(device).detach()
            d_for_q2 = d['q2'].to(device).detach()
            d_for_q3 = d['q3'].to(device).detach()
            d_for_q4 = d['q4'].to(device).detach()
            d_for_q5 = d['q5'].to(device).detach()

            if distillation == True:
                d_for_loss = d_for_q0
            elif distillation == False:
                d_for_loss = d_for_original

            index_divide = 0

            # REAR performance
            out_net_addon_rear_q1 = model_addon_rear_q1(d_for_q1, index_channel=index_divide)
            out_net_addon_rear_q2 = model_addon_rear_q2(d_for_q2, index_channel=index_divide)
            out_net_addon_rear_q3 = model_addon_rear_q3(d_for_q3, index_channel=index_divide)
            out_net_addon_rear_q4 = model_addon_rear_q4(d_for_q4, index_channel=index_divide)
            out_net_addon_rear_q5 = model_addon_rear_q5(d_for_q5, index_channel=index_divide)

            out_criterion_rear_q1 = criterion[index_divide](out_net_addon_rear_q1, d_for_loss)
            out_criterion_rear_q2 = criterion[index_divide](out_net_addon_rear_q2, d_for_loss)
            out_criterion_rear_q3 = criterion[index_divide](out_net_addon_rear_q3, d_for_loss)
            out_criterion_rear_q4 = criterion[index_divide](out_net_addon_rear_q4, d_for_loss)
            out_criterion_rear_q5 = criterion[index_divide](out_net_addon_rear_q5, d_for_loss)

            loss_this_index_rear_q1 = out_criterion_rear_q1["loss"]
            loss_across_index_rear_q1 = np.expand_dims(loss_this_index_rear_q1.cpu().detach().numpy(), axis=1)
            loss_this_index_rear_q2 = out_criterion_rear_q2["loss"]
            loss_across_index_rear_q2 = np.expand_dims(loss_this_index_rear_q2.cpu().detach().numpy(), axis=1)
            loss_this_index_rear_q3 = out_criterion_rear_q3["loss"]
            loss_across_index_rear_q3 = np.expand_dims(loss_this_index_rear_q3.cpu().detach().numpy(), axis=1)
            loss_this_index_rear_q4 = out_criterion_rear_q4["loss"]
            loss_across_index_rear_q4 = np.expand_dims(loss_this_index_rear_q4.cpu().detach().numpy(), axis=1)
            loss_this_index_rear_q5 = out_criterion_rear_q5["loss"]
            loss_across_index_rear_q5 = np.expand_dims(loss_this_index_rear_q5.cpu().detach().numpy(), axis=1)

            loss_rear_q1[index_divide].update(out_criterion_rear_q1["loss"].mean())
            mse_loss_rear_q1[index_divide].update(out_criterion_rear_q1["mse_loss"].mean())
            msssim_loss_rear_q1[index_divide].update(out_criterion_rear_q1["ms_ssim_loss"].mean())
            psnr_rear_q1[index_divide].update(out_criterion_rear_q1['PSNR'].mean())
            msssim_db_rear_q1[index_divide].update(out_criterion_rear_q1['MS-SSIM-DB'].mean())

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

            loss_rear_q4[index_divide].update(out_criterion_rear_q4["loss"].mean())
            mse_loss_rear_q4[index_divide].update(out_criterion_rear_q4["mse_loss"].mean())
            msssim_loss_rear_q4[index_divide].update(out_criterion_rear_q4["ms_ssim_loss"].mean())
            psnr_rear_q4[index_divide].update(out_criterion_rear_q4['PSNR'].mean())
            msssim_db_rear_q4[index_divide].update(out_criterion_rear_q4['MS-SSIM-DB'].mean())

            loss_rear_q5[index_divide].update(out_criterion_rear_q5["loss"].mean())
            mse_loss_rear_q5[index_divide].update(out_criterion_rear_q5["mse_loss"].mean())
            msssim_loss_rear_q5[index_divide].update(out_criterion_rear_q5["ms_ssim_loss"].mean())
            psnr_rear_q5[index_divide].update(out_criterion_rear_q5['PSNR'].mean())
            msssim_db_rear_q5[index_divide].update(out_criterion_rear_q5['MS-SSIM-DB'].mean())

            if i == 0:
                loss_recoder_rear_q1 = LossRecoder(loss_across_index_rear_q1)
                loss_recoder_rear_q2 = LossRecoder(loss_across_index_rear_q2)
                loss_recoder_rear_q3 = LossRecoder(loss_across_index_rear_q3)
                loss_recoder_rear_q4 = LossRecoder(loss_across_index_rear_q4)
                loss_recoder_rear_q5 = LossRecoder(loss_across_index_rear_q5)
            elif i > 0:
                loss_recoder_rear_q1.update_losses(loss_across_index_rear_q1)
                loss_recoder_rear_q2.update_losses(loss_across_index_rear_q2)
                loss_recoder_rear_q3.update_losses(loss_across_index_rear_q3)
                loss_recoder_rear_q4.update_losses(loss_across_index_rear_q4)
                loss_recoder_rear_q5.update_losses(loss_across_index_rear_q5)


            ## BASE performance
            out_net_base_q1 = {"x_hat": d_for_q1}
            out_net_base_q2 = {"x_hat": d_for_q2}
            out_net_base_q3 = {"x_hat": d_for_q3}
            out_net_base_q4 = {"x_hat": d_for_q4}
            out_net_base_q5 = {"x_hat": d_for_q5}

            out_criterion_base_q1 = criterion[index_divide](out_net_base_q1, d_for_loss)
            out_criterion_base_q2 = criterion[index_divide](out_net_base_q2, d_for_loss)
            out_criterion_base_q3 = criterion[index_divide](out_net_base_q3, d_for_loss)
            out_criterion_base_q4 = criterion[index_divide](out_net_base_q4, d_for_loss)
            out_criterion_base_q5 = criterion[index_divide](out_net_base_q5, d_for_loss)

            loss_this_index_base_q1 = out_criterion_base_q1["loss"]
            loss_across_index_base_q1 = np.expand_dims(loss_this_index_base_q1.cpu().detach().numpy(), axis=1)
            loss_this_index_base_q2 = out_criterion_base_q2["loss"]
            loss_across_index_base_q2 = np.expand_dims(loss_this_index_base_q2.cpu().detach().numpy(), axis=1)
            loss_this_index_base_q3 = out_criterion_base_q3["loss"]
            loss_across_index_base_q3 = np.expand_dims(loss_this_index_base_q3.cpu().detach().numpy(), axis=1)
            loss_this_index_base_q4 = out_criterion_base_q4["loss"]
            loss_across_index_base_q4 = np.expand_dims(loss_this_index_base_q4.cpu().detach().numpy(), axis=1)
            loss_this_index_base_q5 = out_criterion_base_q5["loss"]
            loss_across_index_base_q5 = np.expand_dims(loss_this_index_base_q5.cpu().detach().numpy(), axis=1)

            loss_base_q1[index_divide].update(out_criterion_base_q1["loss"].mean())
            mse_loss_base_q1[index_divide].update(out_criterion_base_q1["mse_loss"].mean())
            msssim_loss_base_q1[index_divide].update(out_criterion_base_q1["ms_ssim_loss"].mean())
            psnr_base_q1[index_divide].update(out_criterion_base_q1['PSNR'].mean())
            msssim_db_base_q1[index_divide].update(out_criterion_base_q1['MS-SSIM-DB'].mean())

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

            loss_base_q4[index_divide].update(out_criterion_base_q4["loss"].mean())
            mse_loss_base_q4[index_divide].update(out_criterion_base_q4["mse_loss"].mean())
            msssim_loss_base_q4[index_divide].update(out_criterion_base_q4["ms_ssim_loss"].mean())
            psnr_base_q4[index_divide].update(out_criterion_base_q4['PSNR'].mean())
            msssim_db_base_q4[index_divide].update(out_criterion_base_q4['MS-SSIM-DB'].mean())

            loss_base_q5[index_divide].update(out_criterion_base_q5["loss"].mean())
            mse_loss_base_q5[index_divide].update(out_criterion_base_q5["mse_loss"].mean())
            msssim_loss_base_q5[index_divide].update(out_criterion_base_q5["ms_ssim_loss"].mean())
            psnr_base_q5[index_divide].update(out_criterion_base_q5['PSNR'].mean())
            msssim_db_base_q5[index_divide].update(out_criterion_base_q5['MS-SSIM-DB'].mean())

            if i == 0:
                loss_recoder_base_q1 = LossRecoder(loss_across_index_base_q1)
                loss_recoder_base_q2 = LossRecoder(loss_across_index_base_q2)
                loss_recoder_base_q3 = LossRecoder(loss_across_index_base_q3)
                loss_recoder_base_q4 = LossRecoder(loss_across_index_base_q4)
                loss_recoder_base_q5 = LossRecoder(loss_across_index_base_q5)
            elif i > 0:
                loss_recoder_base_q1.update_losses(loss_across_index_base_q1)
                loss_recoder_base_q2.update_losses(loss_across_index_base_q2)
                loss_recoder_base_q3.update_losses(loss_across_index_base_q3)
                loss_recoder_base_q4.update_losses(loss_across_index_base_q4)
                loss_recoder_base_q5.update_losses(loss_across_index_base_q5)


    for index_divide in range(len(criterion)):
        print(
            f"Test epoch {epoch}: Average losses: "
        )
        print(
            f"\tQ1 Loss (Rear): {loss_rear_q1[index_divide].avg:.3f} |"
            f"\tMSE loss: {mse_loss_rear_q1[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss: {msssim_loss_rear_q1[index_divide].avg:.4f} |'
            f"\tPSNR: {psnr_rear_q1[index_divide].avg:.2f} |"
            f"\tMS-SSIM(DB): {msssim_db_rear_q1[index_divide].avg:.2f} "
        )
        print(
            f"\t   Loss (Base): {loss_base_q1[index_divide].avg:.3f} |"
            f"\tMSE loss: {mse_loss_base_q1[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss: {msssim_loss_base_q1[index_divide].avg:.4f} |'
            f"\tPSNR: {psnr_base_q1[index_divide].avg:.2f} |"
            f"\tMS-SSIM(DB): {msssim_db_base_q1[index_divide].avg:.2f} "
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

        print(
            f"\tQ4 Loss (Rear): {loss_rear_q4[index_divide].avg:.3f} |"
            f"\tMSE loss: {mse_loss_rear_q4[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss: {msssim_loss_rear_q4[index_divide].avg:.4f} |'
            f"\tPSNR: {psnr_rear_q4[index_divide].avg:.2f} |"
            f"\tMS-SSIM(DB): {msssim_db_rear_q4[index_divide].avg:.2f} "
        )
        print(
            f"\t   Loss (Base): {loss_base_q4[index_divide].avg:.3f} |"
            f"\tMSE loss: {mse_loss_base_q4[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss: {msssim_loss_base_q4[index_divide].avg:.4f} |'
            f"\tPSNR: {psnr_base_q4[index_divide].avg:.2f} |"
            f"\tMS-SSIM(DB): {msssim_db_base_q4[index_divide].avg:.2f} "
        )

        print(
            f"\tQ5 Loss (Rear): {loss_rear_q5[index_divide].avg:.3f} |"
            f"\tMSE loss: {mse_loss_rear_q5[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss: {msssim_loss_rear_q5[index_divide].avg:.4f} |'
            f"\tPSNR: {psnr_rear_q5[index_divide].avg:.2f} |"
            f"\tMS-SSIM(DB): {msssim_db_rear_q5[index_divide].avg:.2f} "
        )
        print(
            f"\t   Loss (Base): {loss_base_q5[index_divide].avg:.3f} |"
            f"\tMSE loss: {mse_loss_base_q5[index_divide].avg:.3f} |"
            f'\tMS-SSIM loss: {msssim_loss_base_q5[index_divide].avg:.4f} |'
            f"\tPSNR: {psnr_base_q5[index_divide].avg:.2f} |"
            f"\tMS-SSIM(DB): {msssim_db_base_q5[index_divide].avg:.2f} "
        )

    loss_recoder_rear_q1.update_overall_loss(loss_weights)
    loss_recoder_rear_q2.update_overall_loss(loss_weights)
    loss_recoder_rear_q3.update_overall_loss(loss_weights)
    loss_recoder_rear_q4.update_overall_loss(loss_weights)
    loss_recoder_rear_q5.update_overall_loss(loss_weights)

    return loss_recoder_rear_q1.losses, loss_recoder_rear_q1.overall_loss,\
           loss_recoder_rear_q2.losses, loss_recoder_rear_q2.overall_loss,\
           loss_recoder_rear_q3.losses, loss_recoder_rear_q3.overall_loss,\
           loss_recoder_rear_q4.losses, loss_recoder_rear_q4.overall_loss,\
           loss_recoder_rear_q5.losses, loss_recoder_rear_q5.overall_loss

def save_checkpoint(state, is_best, filedir="", filename="checkpoint"):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filedir + '/best_loss.pth.tar')