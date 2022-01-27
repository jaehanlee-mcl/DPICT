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
import struct
import sys
import time
import math
import numpy as np
import pandas as pd
import os
import warnings

from pathlib import Path
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

import compressai

from compressai.zoo import models, load_state_dict
from compressai.zoo.image import model_architectures as architectures
from pytorch_msssim import ms_ssim
torch.backends.cudnn.deterministic = True
from update_model import update

warnings.filterwarnings(action='ignore')

model_ids = {k: i for i, k in enumerate(models.keys())}

metric_ids = {
    "mse": 0,
}


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


def get_header(model_name, metric, quality):
    """Format header information:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    metric = metric_ids[metric]
    code = (metric << 4) | (quality - 1 & 0x0F)
    return model_ids[model_name], code


def parse_header(header):
    """Read header information from 2 bytes:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    model_id, code = header
    quality = (code & 0x0F) + 1
    metric = code >> 4
    return (
        inverse_dict(model_ids)[model_id],
        inverse_dict(metric_ids)[metric],
        quality,
    )


def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )

def _encode(info, enc_t_mean):
    image = info["image"]
    model = info["architecture"]
    shared = [0/1, 1/1]
    specific = [1/1, 1/1]
    checkpoint = info["updated-path"]
    device = info["device"]
    compressai.set_entropy_coder('ans')

    img = load_image(image)
    net = models[model](N=info["N-Main"], shared_ratio=shared, specific_ratios=specific)
    if checkpoint:
        state_dict = load_state_dict(torch.load(checkpoint, map_location=device))
        net = net.from_state_dict(state_dict, shared_ratio=shared, specific_ratios=specific).eval().to(device)

    x = img2torch(img).to(device)
    p = 64  # maximum 6 strides of 2
    x = pad(x, p)

    enc_start = time.time()
    with torch.no_grad():
        z_strings, z_shape, y, means_hat, scales_hat = net.compress_to_representation(x)
        y_strings = net.compress_to_bitstream(y, means_hat, scales_hat)

    enc_time = time.time() - enc_start
    print(f"Enc {enc_time:.1f}sec, ", end="\t")
    enc_t_mean += enc_time
    return x, y_strings, z_strings, z_shape, enc_t_mean

def save_bitstreams(info, y_strings, z_strings, z_shape, save_bitstreams_t_mean):
    output = info["output_encoded"]
    save_bitstreams_start = time.time()

    if not os.path.exists(output[:-8]):
        os.mkdir(output[:-8])

    with open(output[:-8] + f"/z.bin", "wb") as f:
        write_uints(f, (z_shape[0], z_shape[1]))
        f.write(z_strings[0])

    max_L = len(y_strings) - 1
    pre_indexing = 0

    for i, code in enumerate(y_strings):

        if len(code) == 1:
            with open(output[:-8] + f"/{pre_indexing:03d}_q{max_L - i:02d}.bin", "wb") as f:
                f.write(code[0])
            pre_indexing += 1
        else:
            for j, subcode in enumerate(code):
                with open(output[:-8] + f"/{pre_indexing:03d}_q{max_L - i:02d}_{j + 1:03d}.bin", "wb") as f:
                    f.write(subcode)
                pre_indexing += 1

    save_bitstreams_time = time.time() - save_bitstreams_start
    print(f"Save Bitstreams {save_bitstreams_time:.1f}sec, ", end="\t")
    save_bitstreams_t_mean += save_bitstreams_time

    return save_bitstreams_t_mean

def _decode(info, dec_t_mean):
    compressai.set_entropy_coder("ans")
    inputpath = info["output_encoded"][:-8]
    model = info["architecture"]
    device = info["device"]
    shared = [0/1, 1/1]
    specific = [1/1, 1/1]
    checkpoint = info["updated-path"]

    if not os.path.exists(info["output_decoded"][:-4]):
        os.mkdir(info["output_decoded"][:-4])

    dec_start = time.time()

    start = time.time()
    net = models[model](N=info["N-Main"], shared_ratio=shared, specific_ratios=specific)
    if checkpoint:
        state_dict = load_state_dict(torch.load(checkpoint, map_location=device))
        net = net.from_state_dict(state_dict, shared_ratio=shared, specific_ratios=specific).to(device).eval()

    load_time = time.time() - start

    y_strings_list = os.listdir(inputpath)
    y_strings_list.remove('z.bin')
    y_strings_list.sort()
    for i in range(len(y_strings_list)):
        y_strings_list[i] = inputpath + "/" + y_strings_list[i]

    z_strings_list = open(inputpath + "/" + 'z.bin', "rb")
    z_shape = read_uints(z_strings_list, 2)
    z_strings = z_strings_list.read()

    with torch.no_grad():
        y_hats = net.decompress_to_representation(y_strings_list, z_strings, z_shape)
        x_hats = net.decompress_to_image(y_hats)

    dec_time = time.time() - dec_start
    dec_t_mean += dec_time

    print(f"Dec {dec_time:.2f}sec (model loading: {load_time:.2f}s), ", end="\t")

    return x_hats, y_strings_list, z_strings_list, dec_t_mean

def get_metrics(x_in, x_hats, y_strings_list, z_strings_list, get_metrics_t_mean):
    bpps = []
    psnrs = []
    ssims = []
    get_metrics_start = time.time()

    z_strings = z_strings_list.read()
    bpp = 8 * len(z_strings)
    for index in range(len(x_hats)):
        x_hat = x_hats[index]
        # bpp, psnr, ms-ssim
        bpp += 8 * len(open(y_strings_list[index], "rb").read())
        bpps.append(bpp)
        psnrs.append(-10 * math.log10(F.mse_loss(x_in, x_hat).item()))
        ssims.append(ms_ssim(x_in, x_hat, data_range=1.0).item())

    bpps = np.array(bpps) / (x_in.size(2) * x_in.size(3))

    get_metrics_time = time.time() - get_metrics_start
    get_metrics_t_mean += get_metrics_time
    print(f"Get Metrics {get_metrics_time:.2f}sec, ", end="\t")

    return psnrs, ssims, bpps, get_metrics_t_mean

def save_images(x_hats, y_strings_list, dir_name, save_images_t_mean):
    save_images_start = time.time()

    for index in range(len(x_hats)):
        x_hat = x_hats[index]
        # save image
        ToPILImage()(x_hat.squeeze()).save(dir_name + "/" + "/".join(y_strings_list[index].split("/")[1:])[:-3] + "png")

    save_images_time = time.time() - save_images_start
    save_images_t_mean += save_images_time
    print(f"Save Images {save_images_time:.2f}sec, ", end="\t")

    return save_images_t_mean

def _postprocessing(info, x_hats, postprocessing_t_mean):
    model = info["architecture-post"]
    shared = [0/1, 1/1]
    specific = [1/1, 1/1]
    checkpoint_post2 = info["updated-path-post2"]
    checkpoint_post3 = info["updated-path-post3"]
    device = info["device"]
    postprocessing_start = time.time()

    if not os.path.exists(info["output_post2"][:-4]):
        os.mkdir(info["output_post2"][:-4])
    if not os.path.exists(info["output_post3"][:-4]):
        os.mkdir(info["output_post3"][:-4])

    net_post2 = models[model](N=info["N-Post"], shared_ratio=shared, specific_ratios=specific).to(device)
    if checkpoint_post2:
        net_post2.load_state_dict(torch.load(checkpoint_post2)['state_dict'], pass_update=True)

    net_post3 = models[model](N=info["N-Post"], shared_ratio=shared, specific_ratios=specific).to(device)
    if checkpoint_post3:
        net_post3.load_state_dict(torch.load(checkpoint_post3)['state_dict'], pass_update=True)

    x_hats_post2 = []
    x_hats_post3 = []
    with torch.no_grad():
        for index in range(len(x_hats)):
            if index == 0:
                x_hat_all = x_hats[index]
            else:
                x_hat_all = torch.cat((x_hat_all, x_hats[index]), dim=0)
        x_hat_post2_all = net_post2(x_hat_all)['x_hat']
        x_hat_post3_all = net_post3(x_hat_all)['x_hat']
        for index in range(len(x_hats)):
            x_hats_post2.append(x_hat_post2_all[index:index+1])
            x_hats_post3.append(x_hat_post3_all[index:index+1])

    postprocessing_time = time.time() - postprocessing_start
    print(f"Postprocessing {postprocessing_time:.1f}sec, ", end="\t")
    postprocessing_t_mean += postprocessing_time
    return x_hats_post2, x_hats_post3, postprocessing_t_mean


def _estimation(image, model, metric, quality, coder, show, output=None, get_return=False):
    compressai.set_entropy_coder(coder)
    estimate_start = time.time()

    img = load_image(image)
    start = time.time()
    net = models[model](quality=quality, metric=metric, pretrained=True).eval()
    net.update()
    load_time = time.time() - start

    x = img2torch(img)
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    x = pad(x, p)

    out = net.forward(x)
    x_hat = out['x_hat']
    img = torch2img(x_hat)
    estimate_time = time.time() - estimate_start
    print(f"Estimated in {estimate_time:.2f}s (model loading: {load_time:.2f}s)")

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out["likelihoods"].values()
    )

    if show:
        show_image(img)
    if output is not None:
        img.save(output)

    if get_return == True:
        return x, x_hat, bpp


def show_image(img: Image.Image):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.title.set_text("Decoded image")
    ax.imshow(img)
    fig.tight_layout()
    plt.show()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)

def update_info(info):

    info["filepath"] = info["fileroot"] + f"/{info['epoch-main']:03d}.pth.tar"
    info["updated-name"] = f"updated{info['epoch-main']:03d}"
    info["updated-path"] = info["fileroot"] + f"/{info['updated-name']}.pth.tar"

    info["filepath-post2"] = info["fileroot-post"] + f"/{info['epoch-post']:03d}_2.pth.tar"
    info["updated-name-post2"] = f"{info['epoch-post']:03d}_2"
    info["updated-path-post2"] = info["fileroot-post"] + f"/{info['updated-name-post2']}.pth.tar"

    info["filepath-post3"] = info["fileroot-post"] + f"/{info['epoch-post']:03d}_3.pth.tar"
    info["updated-name-post3"] = f"{info['epoch-post']:03d}_3"
    info["updated-path-post3"] = info["fileroot-post"] + f"/{info['updated-name-post3']}.pth.tar"

    print(f"Model: {info['architecture']:s}, metric: {info['metric']:s},"
          f" checkpoint: {info['filepath']}")
    print("   ")

    if not os.path.exists(f"encoded/{info['experiment']}"):
        os.mkdir(f"encoded/{info['experiment']}")
    if not os.path.exists(f"encoded/{info['experiment']}" + f"/epoch{info['epoch-main']:03d}"):
        os.mkdir(f"encoded/{info['experiment']}" + f"/epoch{info['epoch-main']:03d}")

    if not os.path.exists(f"decoded/{info['experiment']}"):
        os.mkdir(f"decoded/{info['experiment']}")
    if not os.path.exists(f"decoded/{info['experiment']}" + f"/epoch{info['epoch-main']:03d}"):
        os.mkdir(f"decoded/{info['experiment']}" + f"/epoch{info['epoch-main']:03d}")

    if not os.path.exists(f"postprocessing2/{info['experiment']}"):
        os.mkdir(f"postprocessing2/{info['experiment']}")
    if not os.path.exists(f"postprocessing2/{info['experiment']}" + f"/epoch{info['epoch-post']:03d}"):
        os.mkdir(f"postprocessing2/{info['experiment']}" + f"/epoch{info['epoch-post']:03d}")

    if not os.path.exists(f"postprocessing3/{info['experiment']}"):
        os.mkdir(f"postprocessing3/{info['experiment']}")
    if not os.path.exists(f"postprocessing3/{info['experiment']}" + f"/epoch{info['epoch-post']:03d}"):
        os.mkdir(f"postprocessing3/{info['experiment']}" + f"/epoch{info['epoch-post']:03d}")

    info["experiment"] += f"/epoch{info['epoch-main']:03d}-{info['epoch-post']:03d}"

    return info

def update_info(info):

    info["filepath"] = info["fileroot"] + f"/{info['epoch-main']:03d}.pth.tar"
    info["updated-name"] = f"updated{info['epoch-main']:03d}"
    info["updated-path"] = info["fileroot"] + f"/{info['updated-name']}.pth.tar"

    info["filepath-post2"] = info["fileroot-post"] + f"/{info['epoch-post']:03d}_2.pth.tar"
    info["updated-name-post2"] = f"{info['epoch-post']:03d}_2"
    info["updated-path-post2"] = info["fileroot-post"] + f"/{info['updated-name-post2']}.pth.tar"

    info["filepath-post3"] = info["fileroot-post"] + f"/{info['epoch-post']:03d}_3.pth.tar"
    info["updated-name-post3"] = f"{info['epoch-post']:03d}_3"
    info["updated-path-post3"] = info["fileroot-post"] + f"/{info['updated-name-post3']}.pth.tar"

    print(f"Model: {info['architecture']:s}, metric: {info['metric']:s},"
          f" checkpoint: {info['filepath']}")
    print("   ")

    if not os.path.exists(f"encoded/{info['experiment']}"):
        os.mkdir(f"encoded/{info['experiment']}")
    if not os.path.exists(f"encoded/{info['experiment']}" + f"/epoch{info['epoch-main']:03d}"):
        os.mkdir(f"encoded/{info['experiment']}" + f"/epoch{info['epoch-main']:03d}")

    if not os.path.exists(f"decoded/{info['experiment']}"):
        os.mkdir(f"decoded/{info['experiment']}")
    if not os.path.exists(f"decoded/{info['experiment']}" + f"/epoch{info['epoch-main']:03d}"):
        os.mkdir(f"decoded/{info['experiment']}" + f"/epoch{info['epoch-main']:03d}")

    if not os.path.exists(f"postprocessing2/{info['experiment']}"):
        os.mkdir(f"postprocessing2/{info['experiment']}")
    if not os.path.exists(f"postprocessing2/{info['experiment']}" + f"/epoch{info['epoch-post']:03d}"):
        os.mkdir(f"postprocessing2/{info['experiment']}" + f"/epoch{info['epoch-post']:03d}")

    if not os.path.exists(f"postprocessing3/{info['experiment']}"):
        os.mkdir(f"postprocessing3/{info['experiment']}")
    if not os.path.exists(f"postprocessing3/{info['experiment']}" + f"/epoch{info['epoch-post']:03d}"):
        os.mkdir(f"postprocessing3/{info['experiment']}" + f"/epoch{info['epoch-post']:03d}")

    info["experiment"] += f"/epoch{info['epoch-main']:03d}-{info['epoch-post']:03d}"

    return info

def save_record_dict(info, metrics, data_list, data_index_list, dir_name, metric_name):
    record_dict = {}
    for i in range(len(metrics)):
        if metrics[i, 0] == 0:
            end_length = i
            break
    metrics = metrics[:end_length]
    record_dict.setdefault("total bpp", metrics[:, 0])
    record_dict.setdefault("total psnr", metrics[:, 1])
    record_dict.setdefault("total ms-ssim", metrics[:, 2])

    for image_index in range(1, len(data_index_list) + 1):
        record_dict.setdefault(f"data{image_index:02d} bpp", metrics[:, 3 * image_index])
        record_dict.setdefault(f"data{image_index:02d} psnr", metrics[:, 3 * image_index + 1])
        record_dict.setdefault(f"data{image_index:02d} ms-ssim", metrics[:, 3 * image_index + 2])

    data_deepest_index = dir_name + f"/{info['experiment']}" + '/' + data_list[0][:-4]
    deepest_L = 0
    for index in data_index_list:
        if deepest_L < len(os.listdir(dir_name + f"/{info['experiment']}" + '/' + data_list[index][:-4])):
            deepest_L = len(os.listdir(dir_name + f"/{info['experiment']}" + '/' + data_list[index][:-4]))
            data_deepest_index = dir_name + f"/{info['experiment']}" + '/' + data_list[index][:-4]

    recon_image_list = os.listdir(data_deepest_index)
    recon_image_list.sort(reverse=True)
    row_index = [i[4:-4] for i in recon_image_list]

    dataframe = pd.DataFrame(record_dict, index=row_index)
    dataframe.to_csv(metric_name)


def main():
    torch.set_num_threads(1)  # just to be sure
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    ##
    # model: ['bmshj2018-factorized', 'bmshj2018-hyperprior', 'mbt2018-mean', 'mbt2018', 'cheng2020-anchor', 'cheng2020-attn']
    # quality: ([1, 2, 3, 4, 5, 6, 7, 8],)
    # coder: ['ans']

    info = {
        "experiment": "sample",
        "architecture": 'dpict-main',
        "fileroot": 'checkpoint/DPICT-Main',
        "architecture-post": 'dpict-post',
        "fileroot-post": 'checkpoint/DPICT-Post',
        "updated-nameprefix": 'updated',
        "dataset": 'dataset/DPICT-Main/test', #'C:/temp_dataset\compressAI_vimeo_mixture/test_sample',
        "device": "cuda",
        "metric": 'mse',
        "N-Main": 192,
        "N-Post": 192,
        "layer": 10,
        "coder": 'ans',
        "show": False,
        "estimation": False,
        "no-update": False,
        "epoch-main": 0,
        "epoch-post": 0,
    }

    info = update_info(info)
    update(info)

    enc_t_mean = 0
    save_bitstreams_t_mean = 0
    dec_t_mean = 0
    postprocessing_t_mean = 0
    get_metrics_t_mean = 0
    save_images_t_mean = 0
    get_metrics_post2_t_mean = 0
    save_images_post2_t_mean = 0
    get_metrics_post3_t_mean = 0
    save_images_post3_t_mean = 0
    metrics = np.zeros((1000, 3 * 25))
    metric_name = f"decoded/{info['experiment']}/record.csv"
    metrics_post2 = np.zeros((1000, 3 * 25))
    metric_post2_name = f"postprocessing2/{info['experiment']}/record.csv"
    metrics_post3 = np.zeros((1000, 3 * 25))
    metric_post3_name = f"postprocessing3/{info['experiment']}/record.csv"

    data_list = os.listdir(info['dataset'])
    data_index_list = list(range(len(data_list)))

    for index in data_index_list:
        info["image"] = info['dataset'] + '/' + data_list[index]
        if info["estimation"] == True:
            info["output_decoded"] = None
            x_in, x_out, bpp = _estimation(info["image"], info["architecture"], info["metric"], info["quality"], info["coder"], info["show"], info["output_decoded"], get_return=True)

        elif info["estimation"] == False:
            info["output_encoded"] = f"encoded/{info['experiment']}/" + data_list[index] + '.bin'
            info["output_decoded"] = f"decoded/{info['experiment']}/" + data_list[index]
            info["output_post2"] = f"postprocessing2/{info['experiment']}/" + data_list[index]
            info["output_post3"] = f"postprocessing3/{info['experiment']}/" + data_list[index]

            print(f"data{index + 1:02d} ", end="")
            x_in, y_strings, z_strings, z_shape, enc_t_mean = _encode(info, enc_t_mean)
            save_bitstreams_t_mean = save_bitstreams(info, y_strings, z_strings, z_shape, save_bitstreams_t_mean)
            x_hats, y_strings_list, z_strings_list, dec_t_mean = _decode(info, dec_t_mean)
            x_hats_post2, x_hats_post3, postprocessing_t_mean = _postprocessing(info, x_hats, postprocessing_t_mean)
            metric_psnr, metric_ms_ssim, bpp, get_metrics_t_mean = get_metrics(x_in, x_hats, y_strings_list, z_strings_list, get_metrics_t_mean)
            save_images_t_mean = save_images(x_hats, y_strings_list, 'decoded', save_images_t_mean)
            metric_post2_psnr, metric_post2_ms_ssim, bpp, get_metrics_post2_t_mean = get_metrics(x_in, x_hats_post2, y_strings_list, z_strings_list, get_metrics_post2_t_mean)
            save_images_post2_t_mean = save_images(x_hats_post2, y_strings_list, 'postprocessing2', save_images_post2_t_mean)
            metric_post3_psnr, metric_post3_ms_ssim, bpp, get_metrics_post3_t_mean = get_metrics(x_in, x_hats_post3, y_strings_list, z_strings_list, get_metrics_post3_t_mean)
            save_images_post3_t_mean = save_images(x_hats_post3, y_strings_list, 'postprocessing3', save_images_post3_t_mean)
            print('\n')

        metrics[:len(bpp), 3 * (index + 1)] = np.array(bpp[::-1])
        metrics[:len(metric_psnr), 3 * (index + 1) + 1] = np.array(metric_psnr[::-1])
        metrics[:len(metric_ms_ssim), 3 * (index + 1) + 2] = np.array(metric_ms_ssim[::-1])
        metrics[:len(bpp), 0] += np.array(bpp[::-1]) / len(data_index_list)
        metrics[:len(metric_psnr), 1] += np.array(metric_psnr[::-1]) / len(data_index_list)
        metrics[:len(metric_ms_ssim), 2] += np.array(metric_ms_ssim[::-1]) / len(data_index_list)

        metrics_post2[:len(bpp), 3 * (index + 1)] = np.array(bpp[::-1])
        metrics_post2[:len(metric_post2_psnr), 3 * (index + 1) + 1] = np.array(metric_post2_psnr[::-1])
        metrics_post2[:len(metric_post2_ms_ssim), 3 * (index + 1) + 2] = np.array(metric_post2_ms_ssim[::-1])
        metrics_post2[:len(bpp), 0] += np.array(bpp[::-1]) / len(data_index_list)
        metrics_post2[:len(metric_post2_psnr), 1] += np.array(metric_post2_psnr[::-1]) / len(data_index_list)
        metrics_post2[:len(metric_post2_ms_ssim), 2] += np.array(metric_post2_ms_ssim[::-1]) / len(data_index_list)

        metrics_post3[:len(bpp), 3 * (index + 1)] = np.array(bpp[::-1])
        metrics_post3[:len(metric_post3_psnr), 3 * (index + 1) + 1] = np.array(metric_post3_psnr[::-1])
        metrics_post3[:len(metric_post3_ms_ssim), 3 * (index + 1) + 2] = np.array(metric_post3_ms_ssim[::-1])
        metrics_post3[:len(bpp), 0] += np.array(bpp[::-1]) / len(data_index_list)
        metrics_post3[:len(metric_post3_psnr), 1] += np.array(metric_post3_psnr[::-1]) / len(data_index_list)
        metrics_post3[:len(metric_post3_ms_ssim), 2] += np.array(metric_post3_ms_ssim[::-1]) / len(data_index_list)

        dataframe = pd.DataFrame(metrics)
        dataframe.to_csv(metric_name, header=False, index=False)

    print(f"\nAverage Coding time: Enc {enc_t_mean / len(data_index_list):.1f}sec, "
          f" Save Bitstreams {save_bitstreams_t_mean / len(data_index_list):.1f}sec, "
          f" Dec {dec_t_mean / len(data_index_list):.1f}sec, "
          f" Postprocessing {postprocessing_t_mean / len(data_index_list):.1f}sec, "
          f" Get Metrics {get_metrics_t_mean / len(data_index_list):.1f}sec, "
          f" Save Images {save_images_t_mean / len(data_index_list):.1f}sec, "
          f" Get Metrics Post2 {get_metrics_post2_t_mean / len(data_index_list):.1f}sec, "
          f" Save Images Post2 {save_images_post2_t_mean / len(data_index_list):.1f}sec, "
          f" Get Metrics Post3 {get_metrics_post3_t_mean / len(data_index_list):.1f}sec, "
          f" Save Images Post3 {save_images_post3_t_mean / len(data_index_list):.1f}sec, ")

    save_record_dict(info, metrics, data_list, data_index_list, 'decoded', metric_name)
    save_record_dict(info, metrics_post2, data_list, data_index_list, 'postprocessing2', metric_post2_name)
    save_record_dict(info, metrics_post3, data_list, data_index_list, 'postprocessing3', metric_post3_name)
    
if __name__ == "__main__":
    main()
