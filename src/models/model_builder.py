
"""
Author  Yiqun Chen
Docs    Build model from configurations.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn.functional as F
from utils import utils

@utils.log_info_wrapper("Build model from configurations.")
def build_model(cfg, logger=None):
    log_info = print if logger is None else logger.log_info
    raise NotImplementedError("Function build_model do not implemented yet.")