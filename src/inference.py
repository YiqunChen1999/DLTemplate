
r"""
Author:
    Yiqun Chen
Docs:
    Inference.
"""

import os, sys, time
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils import utils, metrics

@torch.no_grad()
def inference(
    cfg, 
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    phase, 
    logger=None, 
    *args, 
    **kwargs, 
):
    model.eval()
    # Prepare to log info.
    log_info = print if logger is None else logger.log_info
    total_loss = []
    inference_time = []
    # Read data and evaluate and record info.
    with utils.log_info(msg="Inference", level="INFO", state=True, logger=logger):
        pbar = tqdm(total=len(data_loader), dynamic_ncols=True)
        for idx, data in enumerate(data_loader):
            start_time = time.time()
            outputs = utils.inference(model=model, data=data, device=device, *args, **kwargs)
            inference_time.append(time.time()-start_time)

            outs = outputs["mask_l"]        

            if cfg.DATA.MULTI_FRAMES:
                outs = outs.reshape(data["mask_s"].shape[0], cfg.DATA.VIDEO.NUM_FRAMES, cfg.DATA.VIDEO.RESOLUTION[0][0], cfg.DATA.VIDEO.RESOLUTION[0][1])

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

            pbar.update()
        pbar.close()
    log_info("Runtime per image: {:<5} seconds.".format(round(sum(inference_time)/len(inference_time), 4)))

