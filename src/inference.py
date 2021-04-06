
r"""
Author:
    Yiqun Chen
Docs:
    Inference.
"""

import os, sys, time, cv2
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
    # Read data and evaluate and record info.
    with utils.log_info(msg="Inference", level="INFO", state=True, logger=logger):
        pbar = tqdm(total=len(data_loader), dynamic_ncols=True)
        for idx, data in enumerate(data_loader):
            outputs, *_ = utils.infer(model=model, data=data, device=device, infer_version=cfg.GENERAL.INFER_VERSION, infer_only=True, *args, **kwargs)

            # Save results to directory.
            for idx_batch in range(outputs.shape[0]):
                out = (outputs[idx_batch].detach().cpu().numpy() * 255).astype(np.uint8)
                dir_save = os.path.join(cfg.SAVE.DIR, data_loader.dataset.dataset, phase)
                utils.try_make_path_exists(dir_save)
                path2dest = os.path.join(dir_save, data["img_idx"][idx_batch]+".png")
                succeed = cv2.imwrite(path2dest, out.transpose(1, 2, 0))
                if not succeed:
                    utils.notify("Failed to save image to {}".format(path2dest))

            pbar.update()
        pbar.close()

