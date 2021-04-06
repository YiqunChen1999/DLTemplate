
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
    metrics_handler, 
    logger=None, 
    *args, 
    **kwargs, 
):
    model.train()
    #  Prepare to log info.
    #  Read data and train and record info.
    data_loader.dataset.update()
    msg = "TRAIN at epoch: {}, lr: {:<5}".format(str(epoch).zfill(3), optimizer.param_groups[0]["lr"])
    with utils.log_info(msg=msg, level="INFO", state=True, logger=logger):
        pbar = tqdm(total=len(data_loader), dynamic_ncols=True)
        for idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            outputs, targets, loss = utils.infer_and_calc_loss(
                model=model, data=data, loss_fn=loss_fn, device=device, infer_version=cfg.GENERAL.INFER_VERSION, *args, **kwargs
            )
            loss.backward()
            optimizer.step()

            cur_loss = loss.detach().cpu().item()
            avg_loss = metrics_handler.update(data_loader.dataset.dataset, "train", epoch, "loss", cur_loss)
            utils.calc_and_record_metrics(data_loader.dataset.dataset, "train", epoch, outputs.detach(), targets.detach(), metrics_handler, 1.0)

            pbar.set_description("Epoch: {:>3} / {:<3}, avg loss: {:<5}, cur loss: {:<5}".format(
                epoch, cfg.TRAIN.MAX_EPOCH, round(avg_loss, 6), round(cur_loss, 6)
            ))
            pbar.update()
        lr_scheduler.step()
        pbar.close()
    metrics_handler.summarize(data_loader.dataset.dataset, "train", epoch, logger=logger)
    return
    # TODO  Return some info.
