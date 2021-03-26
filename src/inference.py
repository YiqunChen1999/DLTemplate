
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
    # Read data and evaluate and record info.
    with utils.log_info(msg="Inference", level="INFO", state=True, logger=logger):
        pbar = tqdm(total=len(data_loader), dynamic_ncols=True)
        for idx, data in enumerate(data_loader):
            outputs = utils.infer(model=model, data=data, device=device, *args, **kwargs)

            # Save results to directory.
            for batch_idx in range(outs.shape[0]):
                # Save results to directory.
                utils.raise_error(NotImplementedError, "Not implemented")            

            pbar.update()
        pbar.close()

