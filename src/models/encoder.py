
r"""
Author:
    Yiqun Chen
Docs:
    Encoder classes.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils
from .modules import *
from utils.logger import logger

ENCODER = {}

def add_encoder(encoder):
    ENCODER[encoder.__name__] = encoder
    return encoder


@add_encoder
class Encoder(nn.Module):
    """"""
    def __init__(self, cfg, *args, **kwargs):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.args = args
        self.kwargs = kwargs
        self._build_()

    def _build_(self):
        logger.raise_error(NotImplementedError, "Encoder is not implemented")
    
    def forward(self, inp, *args, **kwargs):
        logger.raise_error(NotImplementedError, "Encoder is not implemented")
        return out




if __name__ == "__main__":
    print(_ENCODER)