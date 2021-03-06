
r"""
Author:
    Yiqun Chen
Docs:
    Help build loss functions.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from utils import utils

_LOSS_FN = {}

def add_loss_fn(loss_fn):
    _LOSS_FN[loss_fn.__name__] = loss_fn
    return loss_fn

add_loss_fn(torch.nn.MSELoss)


@add_loss_fn
class MSELoss:
    def __init__(self, cfg, *args, **kwargs):
        super(MSELoss, self).__init__()
        self.cfg = cfg
        self.args = args
        self.kwargs = kwargs
        self._build()

    def _build(self):
        self.loss_fn = nn.MSELoss()

    def cal_loss(self, output, target):
        loss = self.loss_fn(output, target)
        return loss

    def __call__(self, output, target):
        return self.cal_loss(output, target)


@add_loss_fn
class MAELoss:
    def __init__(self, cfg, *args, **kwargs):
        super(MAELoss, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.loss_fn = nn.L1Loss()
    
    def cal_loss(self, output, target):
        loss = self.loss_fn(output, target)
        return loss

    def __call__(self, output, target):
        return self.cal_loss(output, target)
        

@add_loss_fn
class MyLossFn:
    def __init__(self, *args, **kwargs):
        super(MyLossFn, self).__init__()
        self._build()

    def _build(self):
        raise NotImplementedError("LossFn is not implemented yet.")

    def cal_loss(self, out, target):
        raise NotImplementedError("LossFn is not implemented yet.")

    def __call__(self, out, target):
        return self.cal_loss(out, target)


def build_loss_fn(cfg, *args, **kwargs):
    return _LOSS_FN[cfg.LOSS_FN.LOSS_FN](cfg, *args, **kwargs)


