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
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import os
import sys
import time
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

import compressai

from compressai.zoo import load_state_dict
from compressai.zoo import models
from compressai.zoo import models as pretrained_models
from compressai.zoo.image import model_architectures as architectures

import struct
from pathlib import Path
from torchvision.transforms import ToPILImage, ToTensor

from update_model import update

torch.backends.cudnn.deterministic = True

model_ids = {k: i for i, k in enumerate(models.keys())}

metric_ids = {
    "mse": 0,
}


torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

def inverse_dict(d):
    # We assume dict values are unique...
    assert len(d.keys()) == len(set(d.keys()))
    return {v: k for k, v in d.items()}


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]

def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)

@torch.no_grad()
def inference_entropy_estimation(model, x, index_slide=0, index_quantize=[0,0,0,0]):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = model.forward(x, index_slide, index_quantize, get_y_hat=True)
    elapsed_time = time.time() - start

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    estimated_bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "decoded": out_net['x_hat'],
        "y_hat": out_net['y_hat'],
        "estimated_bpp": estimated_bpp.item(),
        "estimated_time": elapsed_time,  # broad estimation
    }

def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True
    ).eval()

def load_checkpoint(arch: str, checkpoint_path: str, shared_ratio=[0/8, 5/8], specific_ratios=[5/8, 6/8, 7/8, 8/8]) -> nn.Module:
    state_dict = load_state_dict(torch.load(checkpoint_path))
    return architectures[arch].from_state_dict(state_dict, shared_ratio=shared_ratio, specific_ratios=specific_ratios).eval()

def save_data(model, input_path, output_path, half=False):
    device = next(model.parameters()).device
    if os.path.isdir(output_path) == False:
        os.mkdir(output_path)
    if os.path.isdir(output_path + '/original') == False:
        os.mkdir(output_path + '/original')
    if os.path.isdir(output_path + '/q0') == False:
        os.mkdir(output_path + '/q0')
    if os.path.isdir(output_path + '/q1') == False:
        os.mkdir(output_path + '/q1')
    if os.path.isdir(output_path + '/q2') == False:
        os.mkdir(output_path + '/q2')
    if os.path.isdir(output_path + '/q3') == False:
        os.mkdir(output_path + '/q3')
    if os.path.isdir(output_path + '/q4') == False:
        os.mkdir(output_path + '/q4')
    if os.path.isdir(output_path + '/q5') == False:
        os.mkdir(output_path + '/q5')

    index = 0
    for f in input_path:
        index = index + 1
        print('index:', str(index))
        if index < 0:
            a=0
        else:
            data_name = f.split('\\')[1]
            x = read_image(f).to(device)

            if half:
                model = model.half()
                x = x.half()

            path_decoded = output_path + '/original/' + data_name
            img = torch2img(x)
            img.save(path_decoded)
            for index_quantize in range(6):
                quantize_parameters = [index_quantize, 0, 0, 0]
                estimated_results = inference_entropy_estimation(model, x, index_slide=0, index_quantize=quantize_parameters)
                x_hat = estimated_results['decoded']

                path_decoded = output_path + '/q' + str(index_quantize) + '/' + data_name
                img = torch2img(x_hat)
                img.save(path_decoded)

def main(argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    info = {
        "source": 'checkpoint',
        "experiment": 'make_post_data',
        "updated-nameprefix": 'updated',
        "only-estimation": True,
        "no-update": False,
        "index-slide": 0,
        "N": 192,
        "shared-ratio": [0/1, 1/1],
        "specific-ratios": [1/1, 1/1],
        "index-quantize": [0,0,0,0],

        "architecture": 'dpict-main',
        "metric": 'mse',
        "quality": 1,
        "entropy-coder": 'ans',
        "coding": True,
        "verbose": True,
        "cuda": True,
        "half": False,

        "fileroot": 'checkpoint/DPICT-Main',
        "filepath": 'checkpoint/DPICT-Main/000.pth.tar',
        "updated-name": 'updated000',
        "updated-path": 'checkpoint/DPICT-Main/updated000.pth.tar',

        "input": 'dataset/DPICT-Main',
        "output": 'dataset/DPICT-Post',
    }

    update(info)
    if not info["source"]:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        sys.exit(1)

    inputpath_train = collect_images(info["input"] + '/train')
    if len(inputpath_train) == 0:
        print("Error: no images found in train directory.", file=sys.stderr)
        sys.exit(1)

    inputpath_test = collect_images(info["input"] + '/test')
    if len(inputpath_test) == 0:
        print("Error: no images found in test directory.", file=sys.stderr)
        sys.exit(1)

    if info["source"] == "pretrained":
        runs = sorted([info["quality"]])
        opts = (info["architecture"], info["metric"])
        load_func = load_pretrained
        log_fmt = "\rEvaluating {0} | {run:d}"
    elif info["source"] == "checkpoint":
        runs = [info["updated-path"]]
        opts = ([info["architecture"]])
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s}"

    for run in runs:
        if info["verbose"]:
            sys.stderr.write(log_fmt.format(*opts, run=run))
            sys.stderr.flush()

        if info["source"] == "pretrained":
            model = load_func(*opts, run)
        elif info["source"] == "checkpoint":
            model = load_func(*opts, run, info["shared-ratio"], info["specific-ratios"])

        if info["cuda"] and torch.cuda.is_available():
            model = model.to("cuda")

        save_data(model, inputpath_train, info["output"]+"/train", info["half"])
        save_data(model, inputpath_test, info["output"]+"/test", info["half"])

if __name__ == "__main__":
    main(sys.argv[1:])
