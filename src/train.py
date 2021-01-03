
"""
Author  Yiqun Chen
Docs    Functions to train a model.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch
import torch.nn.functional as F
from utils import utils

@utils.log_info_wrapper("Start train model.")
@torch.no_grad()
def train_one_epoch(
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    loss_fn, 
    optimizer: torch.optim.Optimizer, 
    lr_scheduler, 
    logger=None, 
    *args, 
    **kwargs, 
):
    model.train()
    # TODO  Prepare to log info.
    log_info = print if logger is None else logger.log_info
    # TODO  Read data and train and record info.
    # TODO  Return some info.
    raise NotImplementedError("Function train_one_epoch does not implemented yet.")
