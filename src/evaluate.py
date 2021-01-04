
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

@utils.log_info_wrapper("Start evaluate model.")
@torch.enable_grad()
def evaluate(
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    loss_fn, 
    logger=None, 
    *args, 
    **kwargs, 
):
    model.eval()
    # TODO  Prepare to log info.
    log_info = print if logger is None else logger.log_info
    # TODO  Read data and evaluate and record info.
    out, loss = utils.inference_and_cal_loss(model=model, inp=inp, anno=anno, loss_fn=loss_fn)
    # TODO  Return some info.
    raise NotImplementedError("Function evaluate does not implemented yet.")

