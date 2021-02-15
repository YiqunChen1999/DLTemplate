
r"""
Author:
    Yiqun Chen
Docs:
    Dataset classes.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

from utils import utils

_DATASET = {}

def add_dataset(dataset):
    _DATASET[dataset.__name__] = dataset
    return dataset


@add_dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split, *args, **kwargs):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self._build()

    def _build(self):
        raise NotImplementedError("Dataset is not implemeted yet.")

    def __len__(self):
        raise NotImplementedError("Dataset is not implemeted yet.")

    def __getitem__(self, idx):
        raise NotImplementedError("Dataset is not implemeted yet.")