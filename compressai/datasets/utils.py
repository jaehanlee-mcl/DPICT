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

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train", level=1):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        if level == 1:
            self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        elif level == 2:
            self.dirs = [f for f in splitdir.iterdir() if f.is_dir()]
            self.samples = []
            for index_dir in range(len(self.dirs)):
                temp_samples = [f for f in self.dirs[index_dir].iterdir() if f.is_file()]
                self.samples.extend(temp_samples)

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

class ImageFolderAddOn(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None):
        splitdir = Path(root)

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.dirs = [f for f in splitdir.iterdir() if f.is_dir()]

        self.samples_addon = []
        for index_dir in range(len(self.dirs)):
            if index_dir == 0:
                self.samples = [f for f in self.dirs[index_dir].iterdir() if f.is_file()]
            elif index_dir > 0:
                temp_samples = [f for f in self.dirs[index_dir].iterdir() if f.is_file()]
                self.samples_addon.append(temp_samples)

        self.transform = transform
        self.num_dir = len(self.dirs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        images = []
        for index_dir in range(self.num_dir):
            if index_dir == 0:
                img = Image.open(self.samples[index]).convert("RGB")
            elif index_dir in [1,3,4]: # only use q0, q2, q3
                img = Image.open(self.samples_addon[index_dir-1][index]).convert("RGB")
            images.append(img)

        sample = {'original': images[0], 'q0': images[1], 'q1': images[2], 'q2': images[3], 'q3': images[4], 'q4': images[5], 'q5': images[6]}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
