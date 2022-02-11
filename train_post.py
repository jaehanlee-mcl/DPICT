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

from utils import *

import argparse
import random
import sys
import os
import logging
import logging.handlers

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder, ImageFolderAddOn
from compressai.zoo import models
import data_post

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--dir", type=str, default='DPICT-Main', help="filename",)
    parser.add_argument("-m", "--model", default="dpict-main", choices=models.keys(), help="Model architecture (default: %(default)s)",)
    parser.add_argument("--dataset_test", type=str, default='dataset/DPICT-Post/test', help="Training dataset",)
    parser.add_argument("--dataset_train", type=str, default='dataset/DPICT-Post/train', help="Training dataset",)
    parser.add_argument("--dataset_level", type=int, default=1, help="Directory hierarchy",)
    parser.add_argument("-e", "--epochs", default=500, type=int, help="Number of epochs (default: %(default)s)",)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)",)
    parser.add_argument("-n", "--num-workers", type=int, default=16, help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--shared_ratio", type=list, default=[0/1, 1/1], help="divide parameters",)
    parser.add_argument("--specific_ratios", type=list, default=[1/1, 1/1], help="divide parameters",)
    parser.add_argument("--quantize-parameters", type=list, default=[0], help="quantize parameters",) # [-1, -0.5, 0, 0.5, 1], [-4, -2, 0, 2, 4]
    parser.add_argument("--quantize-randomize-parameters", type=list, default=[0.0], help="quantize parameters",)
    parser.add_argument("--lambda", dest="lmbda", type=list, default=[0.0500], help="Bit-rate distortion parameter (default: %(default)s)",)
    parser.add_argument("--initial_loss_weights", type=list, default=[1], help="initial loss weights",)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size", type=int, default=8, help="Test batch size (default: %(default)s)",)
    parser.add_argument("--aux-learning-rate", default=1e-3, help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--seed", type=float, help="Set random seed for reproducibility")
    parser.add_argument("--clip_max_norm", default=1.0, type=float, help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--checkpoint", type=str, default='checkpoint/DPICT-Main/000.pth.tar', help="Path to a checkpoint") # 'checkpoint/bmshj2018-factorized-lambda0.01/199.pth.tar'
    parser.add_argument("--checkpoint-only-weight", type=bool, default=True)

    parser.add_argument("--N", type=int, default=192, help="network size",)
    parser.add_argument("--dir-addon-rear", type=str, default='DPICT-Post', help="filename",)
    parser.add_argument("--model-addon-rear", default="dpict-post", choices=models.keys(), help="Model architecture (default: %(default)s)",)
    parser.add_argument("--checkpoint-addon-rear-q2", type=str, default=None, help="Path to a checkpoint") # 'checkpoint/bmshj2018-factorized-lambda0.01/199.pth.tar'
    parser.add_argument("--checkpoint-addon-rear-q3", type=str, default=None, help="Path to a checkpoint") # 'checkpoint/bmshj2018-factorized-lambda0.01/199.pth.tar'
    parser.add_argument("--checkpoint-addon-rear-only-weight-q2", type=bool, default=False)
    parser.add_argument("--checkpoint-addon-rear-only-weight-q3", type=bool, default=False)
    args = parser.parse_args(argv)
    return args


def main(argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [data_post.minimum_resize(args.patch_size), data_post.RandomCrop(args.patch_size), data_post.RandomHorizontalFlip(), data_post.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [data_post.CenterCrop(args.patch_size), data_post.ToTensor()]
    )

    train_dataset = ImageFolderAddOn(root=args.dataset_train, transform=train_transforms)
    test_dataset = ImageFolderAddOn(root=args.dataset_test, transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net_addon_rear_q2 = models[args.model_addon_rear](N=args.N, shared_ratio=args.shared_ratio, specific_ratios=args.specific_ratios)
    net_addon_rear_q3 = models[args.model_addon_rear](N=args.N, shared_ratio=args.shared_ratio, specific_ratios=args.specific_ratios)

    net_addon_rear_q2 = net_addon_rear_q2.to(device)
    net_addon_rear_q3 = net_addon_rear_q3.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net_addon_rear_q2 = CustomDataParallel(net_addon_rear_q2)
        net_addon_rear_q3 = CustomDataParallel(net_addon_rear_q3)

    optimizer_addon_rear_q2, aux_optimizer_addon_rear_q2 = configure_optimizers(net_addon_rear_q2, args)
    optimizer_addon_rear_q3, aux_optimizer_addon_rear_q3 = configure_optimizers(net_addon_rear_q3, args)

    criterion = []
    for index in range(len(args.lmbda)):
        criterion.append(DistortionLoss())
    loss_weights = np.array(args.initial_loss_weights)
    initial_scales = np.array(args.initial_loss_weights)

    last_epoch = 0
    log_format = "%(levelname)s %(asctime)s - %(message)s"
    if args.checkpoint_addon_rear_q2 and args.checkpoint_addon_rear_q3:  # load from previous checkpoint
        logging.basicConfig(filename='log/' + args.dir_addon_rear + '.log',
                            filemode='a',
                            format=log_format,
                            level=logging.DEBUG)

        # loading q2
        logging.info("Loading", args.checkpoint_addon_rear_q2)
        checkpoint_addon_rear = torch.load(args.checkpoint_addon_rear_q2, map_location=device)
        net_addon_rear_q2.load_state_dict(checkpoint_addon_rear["state_dict"])
        if args.checkpoint_addon_rear_only_weight_q2 is False:
            last_epoch = checkpoint_addon_rear["epoch"] + 1
            optimizer_addon_rear_q2.load_state_dict(checkpoint_addon_rear["optimizer"])
            aux_optimizer_addon_rear_q2.load_state_dict(checkpoint_addon_rear["aux_optimizer"])
            loss_weights = checkpoint_addon_rear["loss_weights"]
            initial_scales = checkpoint_addon_rear["initial_scales"]
        # loading q3
        logging.info("Loading", args.checkpoint_addon_rear_q3)
        checkpoint_addon_rear = torch.load(args.checkpoint_addon_rear_q3, map_location=device)
        net_addon_rear_q3.load_state_dict(checkpoint_addon_rear["state_dict"])
        if args.checkpoint_addon_rear_only_weight_q3 is False:
            last_epoch = checkpoint_addon_rear["epoch"] + 1
            optimizer_addon_rear_q3.load_state_dict(checkpoint_addon_rear["optimizer"])
            aux_optimizer_addon_rear_q3.load_state_dict(checkpoint_addon_rear["aux_optimizer"])
            loss_weights = checkpoint_addon_rear["loss_weights"]
            initial_scales = checkpoint_addon_rear["initial_scales"]
    else:
        logging.basicConfig(filename='log/' + args.dir_addon_rear + '.log',
                            filemode='w',
                            format=log_format,
                            level=logging.DEBUG)
    logger = logging.getLogger()

    best_loss_rear_q2 = float("inf")
    best_loss_rear_q3 = float("inf")
    for epoch in range(last_epoch, args.epochs):

        logging.info(f"Learning rate: {optimizer_addon_rear_q2.param_groups[0]['lr']}")
        logging.info(f"Learning rate: {optimizer_addon_rear_q3.param_groups[0]['lr']}")
        for index in range(len(args.lmbda)):
            logging.info(
                f"Shared Ratios: {args.shared_ratio[0]} ~ {args.shared_ratio[1]} | "
                f"Specific Ratios: {args.specific_ratios[index]} ~ {args.specific_ratios[index+1]} | "
                f"Lambda: {args.lmbda[index]} | "
                f"Quantize Parameters: {args.quantize_parameters[index]} | "
                f"Quantize Randomize Parameters: {args.quantize_randomize_parameters[index]} |"
            )
        losses_train_rear_q2, overall_loss_train_rear_q2,\
        losses_train_rear_q3, overall_loss_train_rear_q3,= \
            train_DPICT_post(
                net_addon_rear_q2,
                net_addon_rear_q3,
                criterion,
                train_dataloader,
                optimizer_addon_rear_q2,
                optimizer_addon_rear_q3,
                aux_optimizer_addon_rear_q2,
                aux_optimizer_addon_rear_q3,
                epoch,
                args.learning_rate,
                args.clip_max_norm,
                loss_weights=loss_weights,
            )

        # distillation == True
        logging.info('  ')
        logging.info('  [True distillation]')
        losses_test_rear_q2, overall_loss_test_rear_q2,\
        losses_test_rear_q3, overall_loss_test_rear_q3,\
            = test_DPICT_post(epoch, test_dataloader, net_addon_rear_q2, net_addon_rear_q3,
                                criterion, loss_weights=loss_weights, distillation=True)
        # distillation == False
        logging.info('  [False distillation]')
        test_DPICT_post(epoch, test_dataloader, net_addon_rear_q2, net_addon_rear_q3,
                            criterion, loss_weights=loss_weights, distillation=False)

        is_best_rear_q2 = np.mean(overall_loss_test_rear_q2) < best_loss_rear_q2
        best_loss_rear_q2 = min(np.mean(overall_loss_test_rear_q2), best_loss_rear_q2)
        is_best_rear_q3 = np.mean(overall_loss_test_rear_q3) < best_loss_rear_q3
        best_loss_rear_q3 = min(np.mean(overall_loss_test_rear_q3), best_loss_rear_q3)

        if args.save:
            # q2
            filedir = 'checkpoint/' + args.dir_addon_rear
            if os.path.isdir(filedir) == False:
                os.mkdir(filedir)
            filename = 'checkpoint/' + args.dir_addon_rear + '/' + str(epoch).zfill(3) + '_2'
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net_addon_rear_q2.state_dict(),
                    "losses_train": losses_train_rear_q2,
                    "overall_loss_train": overall_loss_train_rear_q2,
                    "losses_test": losses_test_rear_q2,
                    "overall_loss_test": overall_loss_test_rear_q2,
                    "optimizer": optimizer_addon_rear_q2.state_dict(),
                    "aux_optimizer": aux_optimizer_addon_rear_q2.state_dict(),
                    "loss_weights": loss_weights,
                    "initial_scales": initial_scales,
                },
                is_best_rear_q2,
                filedir=filedir,
                filename=filename,
                suffix="_2",
            )

            # q3
            filedir = 'checkpoint/' + args.dir_addon_rear
            if os.path.isdir(filedir) == False:
                os.mkdir(filedir)
            filename = 'checkpoint/' + args.dir_addon_rear + '/' + str(epoch).zfill(3) + '_3'
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net_addon_rear_q3.state_dict(),
                    "losses_train": losses_train_rear_q3,
                    "overall_loss_train": overall_loss_train_rear_q3,
                    "losses_test": losses_test_rear_q3,
                    "overall_loss_test": overall_loss_test_rear_q3,
                    "optimizer": optimizer_addon_rear_q3.state_dict(),
                    "aux_optimizer": aux_optimizer_addon_rear_q3.state_dict(),
                    "loss_weights": loss_weights,
                    "initial_scales": initial_scales,
                },
                is_best_rear_q3,
                filedir=filedir,
                filename=filename,
                suffix="_3",
            )

if __name__ == "__main__":
    main(sys.argv[1:])
