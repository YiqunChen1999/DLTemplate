
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
from .encoder import ENCODER
from .decoder import DECODER


class Model(nn.Module):
    """
    The Language Guided Video Object Segmentation model
    """
    def __init__(self, cfg, *args, **kwargs):
        super(Model, self).__init__()
        self.cfg = cfg
        self._build_model()
    
    def _build_model(self):
        self.encoder = ENCODER[self.cfg.model.enc.arch](self.cfg)
        self.decoder = DECODER[self.cfg.model.dec.arch](self.cfg)

    def forward(self, high_reso, low_reso, *args, **kwargs):
        bigrid = self.encoder(low_reso)
        out = self.decoder(high_reso, bigrid)
        return out



def build_model(cfg, logger=None):
    with utils.log_info(msg="Build model from configurations.", state=True, logger=logger):
        model = Model(cfg)
    return model