from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from io import BytesIO
from PIL import Image
import torch
import random
from torchvision.transforms import functional as F
import math
import random

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class minimum_resize(object):
    def __init__(self, patch_size):
        self.patch_height = patch_size[0]
        self.patch_width = patch_size[1]

    def __call__(self, sample):
        original, q0, q1, q2, q3, q4, q5 = sample['original'], sample['q0'], sample['q1'], sample['q2'], sample['q3'], sample['q4'], sample['q5']

        width = original.size[0]
        height = original.size[1]
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
            original = original.resize(size=[width, height])
            q0 = q0.resize(size=[width, height])
            q1 = q1.resize(size=[width, height])
            q2 = q2.resize(size=[width, height])
            q3 = q3.resize(size=[width, height])
            q4 = q4.resize(size=[width, height])
            q5 = q5.resize(size=[width, height])

        return {'original': original, 'q0': q0, 'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4, 'q5': q5}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        original, q0, q1, q2, q3, q4, q5 = sample['original'], sample['q0'], sample['q1'], sample['q2'], sample['q3'], sample['q4'], sample['q5']

        if not _is_pil_image(original):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(original)))

        if random.random() < 0.5:
            original = original.transpose(Image.FLIP_LEFT_RIGHT)
            q0 = q0.transpose(Image.FLIP_LEFT_RIGHT)
            q1 = q1.transpose(Image.FLIP_LEFT_RIGHT)
            q2 = q2.transpose(Image.FLIP_LEFT_RIGHT)
            q3 = q3.transpose(Image.FLIP_LEFT_RIGHT)
            q4 = q4.transpose(Image.FLIP_LEFT_RIGHT)
            q5 = q5.transpose(Image.FLIP_LEFT_RIGHT)

        return {'original': original, 'q0': q0, 'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4, 'q5': q5}

class RandomCrop(object):

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        original, q0, q1, q2, q3, q4, q5 = sample['original'], sample['q0'], sample['q1'], sample['q2'], sample['q3'], sample['q4'], sample['q5']
        i, j, h, w = self.get_params(original, self.patch_size)

        original = F.crop(original, i, j, h, w)
        q0 = F.crop(q0, i, j, h, w)
        q1 = F.crop(q1, i, j, h, w)
        q2 = F.crop(q2, i, j, h, w)
        q3 = F.crop(q3, i, j, h, w)
        q4 = F.crop(q4, i, j, h, w)
        q5 = F.crop(q5, i, j, h, w)
        return {'original': original, 'q0': q0, 'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4, 'q5': q5}

    def get_params(self, img, output_size):
        w, h = F.get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

class CenterCrop(object):

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, sample):
        original, q0, q1, q2, q3, q4, q5 = sample['original'], sample['q0'], sample['q1'], sample['q2'], sample['q3'], sample['q4'], sample['q5']
        original = F.center_crop(original, self.patch_size)
        q0 = F.center_crop(q0, self.patch_size)
        q1 = F.center_crop(q1, self.patch_size)
        q2 = F.center_crop(q2, self.patch_size)
        q3 = F.center_crop(q3, self.patch_size)
        q4 = F.center_crop(q4, self.patch_size)
        q5 = F.center_crop(q5, self.patch_size)

        return {'original': original, 'q0': q0, 'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4, 'q5': q5}

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class ToTensor(object):
    def __init__(self, ratio=1):
        self.ratio = ratio

    def __call__(self, sample):
        original, q0, q1, q2, q3, q4, q5 = sample['original'], sample['q0'], sample['q1'], sample['q2'], sample['q3'], sample['q4'], sample['q5']

        original = self.to_tensor(original)
        q0 = self.to_tensor(q0)
        q1 = self.to_tensor(q1)
        q2 = self.to_tensor(q2)
        q3 = self.to_tensor(q3)
        q4 = self.to_tensor(q4)
        q5 = self.to_tensor(q5)

        return {'original': original, 'q0': q0, 'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4, 'q5': q5}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img