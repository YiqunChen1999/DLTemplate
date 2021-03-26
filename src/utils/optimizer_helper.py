
r"""
Author:
    Yiqun Chen
Docs:
    Help build optimizer.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from . import utils

_OPTIMIZER = {}

def add_optimizer(optim_func):
    _OPTIMIZER[optim_func.__name__] = optim_func
    return optim_func

@add_optimizer
def SGD(cfg, model):
    lr = cfg.OPTIMIZER.SGD.LR
    momentum = cfg.OPTIMIZER.SGD.MOMENTUM if hasattr(cfg.OPTIMIZER.SGD, "MOMENTUM") else 0
    dampening = cfg.OPTIMIZER.SGD.DAMPENING if hasattr(cfg.OPTIMIZER.SGD, "DAMPENING") else 0
    weight_decay = cfg.OPTIMIZER.SGD.WEIGHT_DECAY if hasattr(cfg.OPTIMIZER.SGD, "WEIGHT_DECAY") else 0
    nesterov = cfg.OPTIMIZER.SGD.NESTEROV if hasattr(cfg.OPTIMIZER.SGD, "NESTEROV") else False
    finetune = cfg.OPTIMIZER.SGD.FINETUNE if hasattr(cfg.TRAIN.OPTIMIZER.SGD, "FINETUNE") else 1.0
    optimizer = torch.optim.SGD(
        [
            {"params": model.encoder.parameters(), "lr": lr * finetune}, 
            {"params": model.decoder.parameters(), "lr": lr}, 
        ], 
        lr=lr, 
        momentum=momentum, 
        dampening=dampening, 
        weight_decay=weight_decay, 
        nesterov=nesterov, 
    )
    utils.raise_error(NotImplementedError, "Optimizer SGD is not implemented yet.")
    return optimizer


@add_optimizer
def Adam(cfg, model):
    lr = cfg.OPTIMIZER.Adam.LR
    weight_decay = cfg.OPTIMIZER.Adam.WEIGHT_DECAY if hasattr(cfg.OPTIMIZER.Adam, "WEIGHT_DECAY") else 0
    betas = cfg.OPTIMIZER.Adam.BETAS if hasattr(cfg.OPTIMIZER.Adam, "BETAS") else (0.9, 0.999)
    eps = cfg.OPTIMIZER.Adam.EPS if hasattr(cfg.OPTIMIZER.Adam, "EPS") else 1E-8
    amsgrad = cfg.OPTIMIZER.Adam.AMSGRAD if hasattr(cfg.OPTIMIZER.Adam, "AMSGRAD") else False
    finetune = cfg.OPTIMIZER.Adam.FINETUNE if hasattr(cfg.OPTIMIZER.Adam, "FINETUNE") else 1.0
    if hasattr(model, "device_ids"):
        optimizer = torch.optim.Adam(
            [
                {"params": model.module.encoder.parameters(), "lr": lr * finetune},  
                {'params': model.module.decoder.parameters()}, 
            ], 
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": model.encoder.parameters(), "lr": lr * finetune}, 
                {'params': model.decoder.parameters()}, 
            ], 
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
    utils.raise_error(NotImplementedError, "Optimizer SGD is not implemented yet.")
    return optimizer


@add_optimizer
def AdamW(cfg, model):
    lr = cfg.OPTIMIZER.AdamW.LR
    weight_decay = cfg.OPTIMIZER.AdamW.WEIGHT_DECAY if hasattr(cfg.OPTIMIZER.AdamW, "WEIGHT_DECAY") else 0.01
    betas = cfg.OPTIMIZER.AdamW.BETAS if hasattr(cfg.OPTIMIZER.AdamW, "BETAS") else (0.9, 0.999)
    eps = cfg.OPTIMIZER.AdamW.EPS if hasattr(cfg.OPTIMIZER.AdamW, "EPS") else 1E-8
    amsgrad = cfg.OPTIMIZER.AdamW.AMSGRAD if hasattr(cfg.OPTIMIZER.AdamW, "AMSGRAD") else False
    finetune = cfg.OPTIMIZER.AdamW.FINETUNE if hasattr(cfg.OPTIMIZER.AdamW, "FINETUNE") else 1.0
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr * finetune}, 
            {"params": model.decoder.parameters(), "lr": lr}, 
        ], 
        lr=lr, 
        betas=betas, 
        eps=eps, 
        weight_decay=weight_decay, 
        amsgrad=amsgrad, 
    )
    utils.raise_error(NotImplementedError, "Optimizer SGD is not implemented yet.")
    return optimizer


def build_optimizer(cfg, model, logger=None, *args, **kwargs):
    if cfg.OPTIMIZER.OPTIMIZER not in _OPTIMIZER.keys():
        utils.raise_error(NotImplementedError, "The expected optimizer %s is not implemented" % cfg.OPTIMIZER.OPTIMIZER)
    if cfg.OPTIMIZER.OPTIMIZER not in cfg.OPTIMIZER.keys():
        utils.raise_error(AttributeError, "Configurations for the expected optimizer %s is required" % cfg.OPTIMIZER.OPTIMIZER)
    with utils.log_info(msg="Build optimizer", level="INFO", state=True, logger=logger):
        optimizer = _OPTIMIZER[cfg.OPTIMIZER.OPTIMIZER](cfg, model)
    return optimizer