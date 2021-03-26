
r"""
Author:
    Yiqun Chen
Docs:
    Help build data loader.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils
from .dataset import _DATASET


def build_dataset(cfg, dataset, split):
    return _DATASET[dataset](cfg, split)
    
    
def build_data_loader(cfg, dataset, split):
    with utils.log_info(msg="Build data loader of {} set for {}".format(split, dataset))
    shuffle = True if split == "train" else False
    num_workers = cfg.DATA.NUM_WORKERS
    dataset = build_dataset(cfg, dataset, split)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=cfg.GENERAL.BATCH_SIZE, 
        shuffle=shuffle, 
        num_workers=num_workers
    )
    return data_loader