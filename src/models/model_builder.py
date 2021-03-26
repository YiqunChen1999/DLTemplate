
r"""
Author:
    Yiqun Chen
Docs:
    Build model from configurations.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
from torch import nn
import torch.nn.functional as F

from utils import utils
from .encoder import _ENCODER
from .decoder import _DECODER


class Model(nn.Module):
    """
    The Language Guided Video Object Segmentation model
    """
    def __init__(self, cfg, split_size=2, *args, **kwargs):
        super(Model, self).__init__()
        self.cfg = cfg
        self.split_size = split_size
        self._build_model()
    
    def _build_model(self):
        self.decoder = _DECODER[self.cfg.MODEL.DECODER.ARCH](self.cfg)
        self.encoder = _ENCODER[self.cfg.MODEL.ENCODER.ARCH](self.cfg)

    def forward(self, inp, *args, **kwargs):
        utils.raise_error(NotImplementedError, "Model is not implemented")
        return out



def build_model(cfg, logger=None):
    with utils.log_info(msg="Build model from configurations.", state=True, logger=logger):
        model = RVOS(cfg, split_size=2)
    return model