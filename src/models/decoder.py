
r"""
Author:
    Yiqun Chen
Docs:
    Decoder classes.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils
from .modules import (
    AsymmCrossAttnV1, 
    AsymmCrossAttnV2, 
    MFCNModuleV2, 
)

DECODER = {}

def add_decoder(decoder):
    DECODER[decoder.__name__] = decoder
    return decoder


@add_decoder
class Decoder(torch.nn.Module):
    """
    This module can deal with multi-task.
    """
    def __init__(self, cfg, *args, **kwargs):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.args = args
        self.kwargs = kwargs
        self._build_()

    def _build_(self):
        utils.raise_error(NotImplementedError, "Decoder is not implemented")

    def forward(self, inp, *args, **kwargs):
        utils.raise_error(NotImplementedError, "Decoder is not implemented")
        return out



if __name__ == "__main__":
    print(_DECODER)
    model = _DECODER["UNetDecoder"](None)
    print(_DECODER)