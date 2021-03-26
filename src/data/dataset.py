
r"""
Author:
    Yiqun Chen
Docs:
    Dataset classes.
"""

import os, sys, cv2, json, copy, h5py, math, random
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from utils import utils

_DATASET = {}

def add_dataset(dataset):
    _DATASET[dataset.__name__] = dataset
    return dataset


@add_dataset
class Dataset(BaseDataset):
    r"""
    Info:
        A dataset requires `_build_` and `__getitem__` methods as well as `items` attribute.
    """
    def __init__(self, cfg, split, *args, **kwargs):
        super(Dataset, self).__init__(cfg, split, "Dataset")
        self.items = []

    def _build_(self):
        utils.raise_error(NotImplementedError, "Dataset is not implemented")

    def __getitem__(self, idx):
        utils.raise_error(NotImplementedError, "Dataset is not implemented")


class BaseDataset(torch.utils.data.Dataset):
    r"""
    Info:
        This is a base dataset.
    """
    def __init__(self, cfg, split, dataset, *args, **kwargs):
        super(BaseDataset, self).__init__()
        if not split in ["train", "valid", "test"]:
            utils.raise_error(ValueError, "Unknown split %s" % split)
        self.cfg = cfg
        self.split = split
        self.dataset = dataset
        self._build_()
        self.update()

    def _build_(self):
        self.items = []
        utils.raise_error(NotImplementedError, "Dataset is not implemeted yet.")
        self.update()

    def update(self):
        if self.split in ["train"]:
            k = int(len(self.items) * self.cfg.DATA[self.dataset].DATA_RATIO)
            random_samples_idx = sorted(random.sample(range(0, len(self.items)), k))
            self.random_indexes = {idx: random_samples_idx[idx] for idx in range(k)}
        elif self.split in ["valid", "test"]:
            self.random_indexes = {idx: idx for idx in range(len(self.items))}

    def __len__(self):
        return len(self.random_indexes)

    def __getitem__(self, idx):
        item = self.items[self.random_indexes[idx]]
        utils.raise_error(NotImplementedError, "Dataset is not implemeted yet.")


if __name__ == "__main__":
    pass




