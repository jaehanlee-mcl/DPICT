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

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--dir", type=str, default='DPICT-Main', help="filename",)
    parser.add_argument("-m", "--model", default="dpict-main", choices=models.keys(), help="Model architecture (default: %(default)s)",)
    parser.add_argument("-d", "--dataset", type=str, default='dataset\DPICT-Main', help="Training dataset",)
    parser.add_argument("--dataset_level", type=int, default=1, help="Directory hierarchy",)
    parser.add_argument("-e", "--epochs", default=500, type=int, help="Number of epochs (default: %(default)s)",)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)",)
    parser.add_argument("-n", "--num-workers", type=int, default=10, help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--N", type=int, default=192, help="network size",)
    parser.add_argument("--lambda", dest="lmbda", type=list, default=[0.2], help="Bit-rate distortion parameter (default: %(default)s)",)
    parser.add_argument("--initial_loss_weights", type=list, default=[1.0], help="initial loss weights",)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size", type=int, default=16, help="Test batch size (default: %(default)s)",)
    parser.add_argument("--aux-learning-rate", default=1e-3, help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--seed", type=float, help="Set random seed for reproducibility")
    parser.add_argument("--clip_max_norm", default=1.0, type=float, help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint") # 'checkpoint/bmshj2018-factorized-lambda0.01/199.pth.tar'
    parser.add_argument("--checkpoint-only-weight", type=bool, default=False)

    parser.add_argument("--shared_ratio", type=list, default=[0/1, 1/1], help="shared parameters",)
    parser.add_argument("--specific_ratios", type=list, default=[1/1, 1/1], help="specific parameters",)
    parser.add_argument("--quantize_parameters", type=list, default=[0], help="quantize parameters",) # [-1, -0.5, 0, 0.5, 1], [-4, -2, 0, 2, 4]
    parser.add_argument("--quantize_randomize_parameters", type=list, default=[0.0], help="quantize parameters",)
    args = parser.parse_args(argv)
    return args


def main(argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [minimum_resize(args.patch_size), transforms.RandomCrop(args.patch_size), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(root=args.dataset, split="train", transform=train_transforms, level=args.dataset_level)
    test_dataset = ImageFolder(root=args.dataset, split="test", transform=test_transforms)

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

    net = models[args.model](N=args.N, shared_ratio=args.shared_ratio, specific_ratios=args.specific_ratios)
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    criterion = []
    for index in range(len(args.lmbda)):
        criterion.append(RateDistortionLoss(lmbda=args.lmbda[index]))
    loss_weights = np.array(args.initial_loss_weights)
    initial_scales = np.array(args.initial_loss_weights)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.checkpoint_only_weight is False:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            loss_weights = checkpoint["loss_weights"]
            initial_scales = checkpoint["initial_scales"]

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):

        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        for index in range(len(args.lmbda)):
            print(
                f"Shared Ratios: {args.shared_ratio[0]} ~ {args.shared_ratio[1]} | "
                f"Specific Ratios: {args.specific_ratios[index]} ~ {args.specific_ratios[index+1]} | "
                f"Lambda: {args.lmbda[index]} | "
                f"Quantize Parameters: {args.quantize_parameters[index]} | "
                f"Quantize Randomize Parameters: {args.quantize_randomize_parameters[index]}"
            )
        losses_train, overall_loss_train = \
            train_DPICT_main(
                net,
                criterion,
                train_dataloader,
                optimizer,
                aux_optimizer,
                epoch,
                args.learning_rate,
                args.clip_max_norm,
                args.quantize_parameters,
                args.quantize_randomize_parameters,
                loss_weights=loss_weights,
            )
        # distillation == True
        print('  ')
        print('  [True distillation]')
        losses_test, overall_loss_test = \
            test_DPICT_main(epoch, test_dataloader, net, criterion, quantize_parameters=args.quantize_parameters, loss_weights=loss_weights, distillation=True)
        # distillation == False
        print('  [False distillation]')
        test_DPICT_main(epoch, test_dataloader, net, criterion, quantize_parameters=args.quantize_parameters, loss_weights=loss_weights, distillation=False)

        is_best = np.mean(overall_loss_test) < best_loss
        best_loss = min(np.mean(overall_loss_test), best_loss)

        net.update()
        if args.save:
            filedir = 'checkpoint/' + args.dir
            if os.path.isdir(filedir) == False:
                os.mkdir(filedir)
            filename = 'checkpoint/' + args.dir + '/' + str(epoch).zfill(3)

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "losses_train": losses_train,
                    "overall_loss_train": overall_loss_train,
                    "losses_test": losses_test,
                    "overall_loss_test": overall_loss_test,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "loss_weights": loss_weights,
                    "initial_scales": initial_scales,
                },
                is_best,
                filedir=filedir,
                filename=filename
            )


if __name__ == "__main__":
    main(sys.argv[1:])
