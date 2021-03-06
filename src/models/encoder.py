
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

_ENCODER = {}

def add_encoder(encoder):
    _ENCODER[encoder.__name__] = encoder
    return encoder


@add_encoder
class Encoder(nn.Module):
    """"""
    def __init__(self, cfg, *args, **kwargs):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.args = args
        self.kwargs = kwargs
        self._build()

    def _build(self):
        raise NotImplementedError("")
    
    def forward(self, frames, text_repr, *args, **kwargs):
        raise NotImplementedError("")
        return video_repr




if __name__ == "__main__":
    print(_ENCODER)