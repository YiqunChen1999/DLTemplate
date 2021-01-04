
"""
Author  Yiqun Chen
Docs    Build model from configurations.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
from torch import nn
import torch.nn.functional as F
from utils import utils
from .encoder import *
from .decoder import *


class Model(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(Model, self).__init__()
        self.cfg = cfg
        self._build_model()

    def _build_model(self):
        self.encoder = _ENCODER[self.cfg.MODEL.ENCODER]
        self.decoder = _DECODER[self.cfg.MODEL.DECODER]
        raise NotImplementedError("Method Model._build_model is not implemented.")

    def forward(self, data, *args, **kwargs):
        raise NotImplementedError("Method Model.forward is not implemented.")


@utils.log_info_wrapper("Build model from configurations.")
def build_model(cfg, logger=None):
    log_info = print if logger is None else logger.log_info
    raise NotImplementedError("Function build_model do not implemented yet.")