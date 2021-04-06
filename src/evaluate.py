
r"""
Author:
    Yiqun Chen
Docs:
    Functions to evaluate a model.
"""

import os, sys, time, cv2
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
            outputs, targets, loss = utils.infer_and_calc_loss(
                model=model, data=data, loss_fn=loss_fn, device=device, infer_version=cfg.GENERAL.INFER_VERSION, *args, **kwargs
            )

            if save:
                # Save results to directory.
                for idx_batch in range(outputs.shape[0]):
                    out = (outputs[idx_batch].detach().cpu().numpy() * 255).astype(np.uint8)
                    dir_save = os.path.join(cfg.SAVE.DIR, data_loader.dataset.dataset, phase)
                    utils.try_make_path_exists(dir_save)
                    path2dest = os.path.join(dir_save, data["img_idx"][idx_batch]+".png")
                    succeed = cv2.imwrite(path2dest, out.transpose(1, 2, 0))
                    if not succeed:
                        utils.notify("Failed to save image to {}".format(path2dest))

            cur_loss = loss.detach().cpu().item()
            avg_loss = metrics_handler.update(data_loader.dataset.dataset, phase, epoch, "loss", cur_loss)
            utils.calc_and_record_metrics(data_loader.dataset.dataset, phase, epoch, outputs, targets, metrics_handler, 1.0)

            pbar.set_description("Epoch: {:>3} / {:<3}, avg loss: {:<5}, cur loss: {:<5}".format(
                epoch, cfg.TRAIN.MAX_EPOCH, round(avg_loss, 6), round(cur_loss, 6)
            ))
            pbar.update()
        pbar.close()
    metrics_handler.summarize(data_loader.dataset.dataset, phase, epoch, logger=logger)
    return
