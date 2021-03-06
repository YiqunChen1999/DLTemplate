
r"""
Author:
    Yiqun Chen
Docs:
    Functions to train a model.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision, random
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import utils

@torch.enable_grad()
def train_one_epoch(
    epoch: int, 
    cfg, 
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    loss_fn, 
    optimizer: torch.optim.Optimizer, 
    lr_scheduler, 
    metrics_logger, 
    logger=None, 
    *args, 
    **kwargs, 
):
    model.train()
    #  Prepare to log info.
    log_info = print if logger is None else logger.log_info
    total_loss = []
    #  Read data and train and record info.
    data_loader.dataset.update()
    with utils.log_info(msg="TRAIN at epoch: {}, lr: {:<5}".format(str(epoch).zfill(3), optimizer.param_groups[0]["lr"]), level="INFO", state=True, logger=logger):
        pbar = tqdm(total=len(data_loader), dynamic_ncols=True)
        for idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            outputs, loss = utils.inference_and_calc_loss(model=model, data=data, loss_fn=loss_fn, device=device, *args, **kwargs)
            loss.backward(); optimizer.step()
            total_loss.append(loss.detach().cpu().item())

            metrics_logger.record("train", epoch, "loss", loss.detach().cpu().item())
            utils.calc_and_record_metrics(
                "train", epoch, outputs, targets, metrics_logger, 
            )

            pbar.set_description("Epoch: {:>3} / {:<3}, avg loss: {:<5}, cur loss: {:<5}".format(epoch, cfg.TRAIN.MAX_EPOCH, round(sum(total_loss)/len(total_loss), 5), round(total_loss[-1], 5)))
            pbar.update()
        lr_scheduler.step()
        pbar.close()
    metrics_logger.summarize("train", log_info)
    return
    # TODO  Return some info.
