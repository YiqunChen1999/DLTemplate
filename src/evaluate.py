
r"""
Author:
    Yiqun Chen
Docs:
    Functions to evaluate a model.
"""

import os, sys, time
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils import utils, metrics

@torch.no_grad()
def evaluate(
    epoch: int, 
    cfg, 
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    loss_fn, 
    metrics_handler, 
    phase="valid", 
    logger=None, 
    save=False, 
    *args, 
    **kwargs, 
):
    model.eval()
    # Read data and evaluate and record info.
    msg="{} at epoch: {}".format(phase.upper(), str(epoch).zfill(3))
    with utils.log_info(msg=msg, level="INFO", state=True, logger=logger):
        pbar = tqdm(total=len(data_loader), dynamic_ncols=True)
        for idx, data in enumerate(data_loader):
            outputs, loss = utils.infer_and_calc_loss(model=model, data=data, loss_fn=loss_fn, device=device, *args, **kwargs)

            if save:
                # Save results to directory.
                utils.raise_error(NotImplementedError, "Not implemented")            

            cur_loss = loss.detach().cpu().item()            
            avg_loss = metrics_handler.record(phase, epoch, "loss", cur_loss)
            utils.calc_and_record_metrics(phase, epoch, outputs, targets, metrics_handler)

            pbar.set_description("Epoch: {:>3} / {:<3}, avg loss: {:<5}, cur loss: {:<5}".format(
                epoch, cfg.TRAIN.MAX_EPOCH, round(avg_loss, 6), round(cur_loss, 6)
            ))
            pbar.update()
        pbar.close()
    metrics_handler.summarize(phase)
    utils.raise_error(NotImplementedError, "Function evaluate is not implemented yet.")

