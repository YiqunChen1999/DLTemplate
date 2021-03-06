
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
    metrics_logger, 
    phase="valid", 
    logger=None, 
    save=False, 
    *args, 
    **kwargs, 
):
    model.eval()
    # Prepare to log info.
    log_info = print if logger is None else logger.log_info
    total_loss = []
    inference_time = []
    # Read data and evaluate and record info.
    with utils.log_info(msg="{} at epoch: {}".format(phase.upper(), str(epoch).zfill(3)), level="INFO", state=True, logger=logger):
        pbar = tqdm(total=len(data_loader), dynamic_ncols=True)
        for idx, data in enumerate(data_loader):
            start_time = time.time()
            outputs, loss = utils.inference_and_calc_loss(model=model, data=data, loss_fn=loss_fn, device=device, *args, **kwargs)
            inference_time.append(time.time()-start_time)
            total_loss.append(loss.detach().cpu().item())

            if save:
                # Save results to directory.
                for batch_idx in range(outs.shape[0]):
                    save_dir = os.path.join(cfg.SAVE.DIR, phase, data["fn_video"][batch_idx])
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    path2file = os.path.join(save_dir, data["frame_idx"][batch_idx]+".png")
                    succeed = utils.save_image(img.numpy().astype(np.uint8), path2file)
                    if not succeed:
                        log_info("Cannot save image to {}".format(path2file))
            
            metrics_logger.record(phase, epoch, "loss", loss.detach().cpu().item())
            utils.calc_and_record_metrics(
                phase, epoch, outs, trgs, metrics_logger, 
                logger=logger
            )

            pbar.set_description("Epoch: {:>3} / {:<3}, avg loss: {:<5}, cur loss: {:<5}".format(epoch, cfg.TRAIN.MAX_EPOCH, round(sum(total_loss)/len(total_loss), 5), round(total_loss[-1], 5)))
            pbar.update()
        pbar.close()
    log_info("Runtime per image: {:<5} seconds.".format(round(sum(inference_time)/len(inference_time), 4)))
    metrics_logger.summarize(phase, log_info)
    # TODO  Return some info.
    raise NotImplementedError("Function evaluate is not implemented yet.")

