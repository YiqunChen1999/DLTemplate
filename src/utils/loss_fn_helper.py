
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
from utils import metrics

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

    def calc_loss(self, output, target):
        loss = self.loss_fn(output, target)
        return loss

    def __call__(self, output, target):
        return self.calc_loss(output, target)


@add_loss_fn
class MAELoss:
    def __init__(self, cfg, *args, **kwargs):
        super(MAELoss, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.loss_fn = nn.L1Loss()
    
    def calc_loss(self, output, target):
        loss = self.loss_fn(output, target)
        return loss

    def __call__(self, output, target):
        return self.calc_loss(output, target)
        

@add_loss_fn
class SSIMLoss:
    def __init__(self, cfg, *args, **kwargs):
        super(SSIMLoss, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.device = torch.device("cpu" if len(self.cfg.GENERAL.GPU) == 0 else "cuda:"+str(self.cfg.GENERAL.GPU[0]))

    def calc_loss(self, output, target):
        loss = 1 - metrics.calc_ssim(output, target, data_range=1.0, multichannel=True, device=self.device)
        return loss

    def __call__(self, output, target):
        return self.calc_loss(output, target)


@add_loss_fn
class MSESSIMLoss:
    def __init__(self, cfg, *args, **kwargs):
        super(MSESSIMLoss, self).__init__()
        self.cfg = cfg
        # self.weights = self.cfg.LOSS_FN.WEIGHTS
        self._build()

    def _build(self):
        self.loss_fn_mse = MSELoss(self.cfg)
        self.loss_fn_ssim = SSIMLoss(self.cfg)
        assert "MSE" in self.cfg.LOSS_FN.MSESSIMLoss.keys() and "SSIM" in self.cfg.LOSS_FN.MSESSIMLoss.keys(), "Weights of loss are not found"

    def calc_loss(self, output, target):
        loss_mse = self.loss_fn_mse(output, target)
        loss_ssim = self.loss_fn_ssim(output, target)
        loss = self.cfg.LOSS_FN.MSESSIMLoss.MSE * loss_mse + self.cfg.LOSS_FN.MSESSIMLoss.SSIM * loss_ssim
        return loss

    def __call__(self, output, target):
        return self.calc_loss(output, target)


@add_loss_fn
class MyLossFn:
    def __init__(self, *args, **kwargs):
        super(MyLossFn, self).__init__()
        self._build()

    def _build(self):
        raise NotImplementedError("LossFn is not implemented yet.")

    def calc_loss(self, out, target):
        raise NotImplementedError("LossFn is not implemented yet.")

    def __call__(self, out, target):
        return self.calc_loss(out, target)


def build_loss_fn(cfg, logger=None, *args, **kwargs):
    with utils.log_info(msg="Build loss function", level="INFO", state=True, logger=logger):
        loss_fn = _LOSS_FN[cfg.LOSS_FN.LOSS_FN](cfg, *args, **kwargs)
    return loss_fn


