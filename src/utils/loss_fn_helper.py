
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
class MaskBCEBBoxMSELoss:
    def __init__(self, cfg, *args, **kwargs):
        super(MaskBCEBBoxMSELoss, self).__init__()
        self.cfg = cfg
        self.weight = self.cfg.LOSS_FN.WEIGHT
        self._build()

    def _build(self):
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def cal_loss(self, output, target):
        
        loss_mask_s = F.binary_cross_entropy_with_logits(output["mask_s"], target["gt_mask_s"], pos_weight=torch.tensor(self.cfg.LOSS_FN.WEIGHTS.POS_WEIGHT))
        loss_mask_m = F.binary_cross_entropy_with_logits(output["mask_m"], target["gt_mask_m"], pos_weight=torch.tensor(self.cfg.LOSS_FN.WEIGHTS.POS_WEIGHT))
        loss_mask_l = F.binary_cross_entropy_with_logits(output["mask_l"], target["gt_mask_l"], pos_weight=torch.tensor(self.cfg.LOSS_FN.WEIGHTS.POS_WEIGHT))
        
        loss_bbox_s = self.mse_loss(output["bbox_s"], target["gt_bbox_s"])
        loss_bbox_m = self.mse_loss(output["bbox_m"], target["gt_bbox_m"])
        loss_bbox_l = self.mse_loss(output["bbox_l"], target["gt_bbox_l"])
        
        loss = loss_mask_s + loss_mask_m + loss_mask_l + \
            self.cfg.LOSS_FN.WEIGHTS.BBOX_COEFF * (loss_bbox_s + loss_bbox_m + loss_bbox_l)

        return loss

    def __call__(self, output, target):
        return self.cal_loss(output, target)


@add_loss_fn
class MaskBCEBBoxMSEMAELoss:
    def __init__(self, cfg, *args, **kwargs):
        super(MaskBCEBBoxMSEMAELoss, self).__init__()
        self.cfg = cfg
        self.weight = self.cfg.LOSS_FN.WEIGHT
        self._build()

    def _build(self):
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def cal_loss(self, output, target):
        
        loss_mask_s = F.binary_cross_entropy_with_logits(output["mask_s"], target["gt_mask_s"], pos_weight=torch.tensor(self.cfg.LOSS_FN.WEIGHTS.POS_WEIGHT))
        loss_mask_m = F.binary_cross_entropy_with_logits(output["mask_m"], target["gt_mask_m"], pos_weight=torch.tensor(self.cfg.LOSS_FN.WEIGHTS.POS_WEIGHT))
        loss_mask_l = F.binary_cross_entropy_with_logits(output["mask_l"], target["gt_mask_l"], pos_weight=torch.tensor(self.cfg.LOSS_FN.WEIGHTS.POS_WEIGHT))
        
        loss_bbox_s = self.mse_loss(output["bbox_s"], target["gt_bbox_s"])
        loss_bbox_m = self.mse_loss(output["bbox_m"], target["gt_bbox_m"])
        loss_bbox_l = self.mse_loss(output["bbox_l"], target["gt_bbox_l"])
        
        mae_loss_bbox_s = self.mae_loss(output["bbox_s"], target["gt_bbox_s"])
        mae_loss_bbox_m = self.mae_loss(output["bbox_m"], target["gt_bbox_m"])
        mae_loss_bbox_l = self.mae_loss(output["bbox_l"], target["gt_bbox_l"])
        
        loss = loss_mask_s + loss_mask_m + loss_mask_l + \
            self.cfg.LOSS_FN.WEIGHTS.BBOX_COEFF * (loss_bbox_s + loss_bbox_m + loss_bbox_l) + \
                self.cfg.LOSS_FN.WEIGHTS.BBOX_COEFF * (mae_loss_bbox_s + mae_loss_bbox_m + mae_loss_bbox_l)

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


