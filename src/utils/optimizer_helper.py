
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

OPTIMIZER = {}

def add_optimizer(optim_func):
    OPTIMIZER[optim_func.__name__] = optim_func
    return optim_func

@add_optimizer
def SGD(cfg, model):
    lr = cfg.optim.SGD.LR
    momentum = cfg.optim.SGD.MOMENTUM if hasattr(cfg.optim.SGD, "MOMENTUM") else 0
    dampening = cfg.optim.SGD.DAMPENING if hasattr(cfg.optim.SGD, "DAMPENING") else 0
    weight_decay = cfg.optim.SGD.WEIGHT_DECAY if hasattr(cfg.optim.SGD, "WEIGHT_DECAY") else 0
    nesterov = cfg.optim.SGD.NESTEROV if hasattr(cfg.optim.SGD, "NESTEROV") else False
    finetune = cfg.optim.SGD.FINETUNE if hasattr(cfg.train.OPTIMIZER.SGD, "FINETUNE") else 1.0
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
    lr = cfg.optim.Adam.lr
    weight_decay = cfg.optim.Adam.weight_decay if hasattr(cfg.optim.Adam, "WEIGHT_DECAY") else 0
    betas = cfg.optim.Adam.BETAS if hasattr(cfg.optim.Adam, "BETAS") else (0.9, 0.999)
    eps = cfg.optim.Adam.EPS if hasattr(cfg.optim.Adam, "EPS") else 1E-8
    amsgrad = cfg.optim.Adam.AMSGRAD if hasattr(cfg.optim.Adam, "AMSGRAD") else False
    finetune = cfg.optim.Adam.finetune if hasattr(cfg.optim.Adam, "FINETUNE") else 1.0
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
    return optimizer


@add_optimizer
def AdamW(cfg, model):
    lr = cfg.optim.AdamW.lr
    weight_decay = cfg.optim.AdamW.weight_decay if hasattr(cfg.optim.AdamW, "WEIGHT_DECAY") else 0.01
    betas = cfg.optim.AdamW.BETAS if hasattr(cfg.optim.AdamW, "BETAS") else (0.9, 0.999)
    eps = cfg.optim.AdamW.EPS if hasattr(cfg.optim.AdamW, "EPS") else 1E-8
    amsgrad = cfg.optim.AdamW.AMSGRAD if hasattr(cfg.optim.AdamW, "AMSGRAD") else False
    finetune = cfg.optim.AdamW.finetune if hasattr(cfg.optim.AdamW, "FINETUNE") else 1.0
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
    if cfg.optim.optim not in _OPTIMIZER.keys():
        utils.raise_error(NotImplementedError, "The expected optimizer %s is not implemented" % cfg.optim.optim)
    if cfg.optim.optim not in cfg.optim.keys():
        utils.raise_error(AttributeError, "Configurations for the expected optimizer %s is required" % cfg.optim.optim)
    with utils.log_info(msg="Build optimizer", level="INFO", state=True, logger=logger):
        optimizer = OPTIMIZER[cfg.optim.optim](cfg, model)
    return optimizer