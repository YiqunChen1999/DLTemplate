
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
            outputs, targets, loss = utils.inference_and_calc_loss(model=model, data=data, loss_fn=loss_fn, device=device, *args, **kwargs)
            inference_time.append(time.time()-start_time)
            total_loss.append(loss.detach().cpu().item())

            outs = outputs["mask_l"]
            trgs = targets["gt_mask_l"]
            if cfg.DATA.MULTI_FRAMES:
                outs = outs.reshape(data["mask_s"].shape[0], cfg.DATA.VIDEO.NUM_FRAMES, cfg.DATA.VIDEO.RESOLUTION[0][0], cfg.DATA.VIDEO.RESOLUTION[0][1])
                trgs = trgs.reshape(data["mask_s"].shape[0], cfg.DATA.VIDEO.NUM_FRAMES, cfg.DATA.VIDEO.RESOLUTION[0][0], cfg.DATA.VIDEO.RESOLUTION[0][1])
            
            if save:
                # Save results to directory.
                for batch_idx in range(outs.shape[0]):
                    save_dir = os.path.join(cfg.SAVE.DIR, phase, data["fn_video"][batch_idx])
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    if cfg.DATA.MULTI_FRAMES:
                        for frame_idx in range(outs.shape[1]):
                            mask = utils.crop_and_resize(
                                outs[batch_idx][frame_idx].detach().cpu(), 
                                (data["padding"][0][batch_idx], data["padding"][1][batch_idx], data["padding"][2][batch_idx], data["padding"][3][batch_idx]), 
                                (data["size"][0][batch_idx], data["size"][1][batch_idx]), 
                                False, 
                            )
                            # mask = ((mask > (torch.max(mask) / 2)) * 255).numpy().astype(np.uint8)
                            path2file = os.path.join(save_dir, data["fns"][frame_idx][batch_idx]+".png")
                            succeed = utils.save_image(mask.numpy().astype(np.uint8), path2file)
                            if not succeed:
                                log_info("Cannot save image to {}".format(path2file))
                    else:
                        mask = utils.crop_and_resize(
                            outs[batch_idx].detach().cpu(), 
                            (data["padding"][0][batch_idx], data["padding"][1][batch_idx], data["padding"][2][batch_idx], data["padding"][3][batch_idx]), 
                            (data["size"][0][batch_idx], data["size"][1][batch_idx]), 
                            False, 
                        )
                        # mask = ((mask > (torch.max(mask) / 2)) * 255).numpy().astype(np.uint8)
                        path2file = os.path.join(save_dir, data["frame_idx"][batch_idx]+".png")
                        succeed = utils.save_image(mask.numpy().astype(np.uint8), path2file)
                        if not succeed:
                            log_info("Cannot save image to {}".format(path2file))
            
            metrics_logger.record(phase, epoch, "loss", loss.detach().cpu().item())
            utils.calc_and_record_metrics(
                phase, epoch, outs, trgs, metrics_logger, 
                multi_frames=cfg.DATA.MULTI_FRAMES, 
                require_resize=False, 
                padding=data["padding"], 
                size=data["size"], 
                logger=logger
            )

            pbar.set_description("Epoch: {:>3} / {:<3}, avg loss: {:<5}, cur loss: {:<5}".format(epoch, cfg.TRAIN.MAX_EPOCH, round(sum(total_loss)/len(total_loss), 5), round(total_loss[-1], 5)))
            pbar.update()
        pbar.close()
    log_info("Runtime per image: {:<5} seconds.".format(round(sum(inference_time)/len(inference_time), 4)))
    mean_metrics = metrics_logger.mean("train", epoch)
    log_info("Jaccard: {:<5}, F Score: {:<5}, Loss: {:<5}".format(
        mean_metrics["Jaccard"], mean_metrics["F_score"], mean_metrics["loss"], 
    ))
    # TODO  Return some info.
    raise NotImplementedError("Function evaluate is not implemented yet.")

