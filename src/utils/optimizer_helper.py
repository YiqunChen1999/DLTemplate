
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

from utils import utils

_OPTIMIZER = {}

def add_optimizer(optim_func):
    _OPTIMIZER[optim_func.__name__] = optim_func
    return optim_func

@add_optimizer
def SGD(cfg, model):
    lr = cfg.OPTIMIZER.LR
    momentum = cfg.OPTIMIZER.MOMENTUM if cfg.OPTIMIZER.hasattr("MOMENTUM") else 0
    dampening = cfg.OPTIMIZER.DAMPENING if cfg.OPTIMIZER.hasattr("DAMPENING") else 0
    weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY if cfg.OPTIMIZER.hasattr("WEIGHT_DECAY") else 0
    nesterov = cfg.OPTIMIZER.NESTEROV if cfg.OPTIMIZER.hasattr("NESTEROV") else False
    optimizer = torch.optim.SGD(
        [
            {"params": model.encoder.parameters(), "lr": lr * cfg.OPTIMIZER.LR_FACTOR}, 
            {"params": model.decoder.parameters(), "lr": lr}, 
            # {"params": model.backbone.parameters(), "lr": lr}, 
        ], 
        lr=lr, 
        momentum=momentum, 
        dampening=dampening, 
        weight_decay=weight_decay, 
        nesterov=nesterov, 
    )
    raise NotImplementedError("Optimizer SGD is not implemented yet.")
    return optimizer


@add_optimizer
def Adam(cfg, model):
    lr = cfg.OPTIMIZER.LR
    weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY if cfg.OPTIMIZER.hasattr("WEIGHT_DECAY") else 0
    betas = cfg.OPTIMIZER.BETAS if cfg.OPTIMIZER.hasattr("BETAS") else (0.9, 0.999)
    eps = cfg.OPTIMIZER.EPS if cfg.OPTIMIZER.hasattr("EPS") else 1E-8
    amsgrad = cfg.OPTIMIZER.AMSGRAD if cfg.OPTIMIZER.hasattr("AMSGRAD") else False
    finetune_lr_factor = cfg.TRAIN.OPTIMIZER.FINETUNE_FACTOR if cfg.TRAIN.OPTIMIZER.hasattr("FINETUNE_FACTOR") else 1.0
    if hasattr(model, "device_ids"):
        optimizer = torch.optim.Adam(
            [
                {'params': model.module.text_encoder.parameters()}, 
                {'params': model.module.decoder.parameters()}, 
                {'params': model.module.video_encoder.model.parameters(), 'lr': lr * finetune_lr_factor}, 
                # {
                #     'params': model.module.word_embedding.embeddings.parameters(), 
                #     'lr': lr * finetune_lr_factor, 
                # },  
            ], 
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {'params': model.text_encoder.parameters()}, 
                {'params': model.decoder.parameters()}, 
                {'params': model.video_encoder.model.parameters(), 'lr': lr * finetune_lr_factor}, 
                # {
                #     'params': model.word_embedding.embeddings.parameters(), 
                #     'lr': lr * finetune_lr_factor, 
                # },  
            ], 
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )

    return optimizer


@add_optimizer
def AdamW(cfg, model):
    lr = cfg.OPTIMIZER.LR
    weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY if cfg.OPTIMIZER.hasattr("WEIGHT_DECAY") else 0.01
    betas = cfg.OPTIMIZER.BETAS if cfg.OPTIMIZER.hasattr("BETAS") else (0.9, 0.999)
    eps = cfg.OPTIMIZER.EPS if cfg.OPTIMIZER.hasattr("EPS") else 1E-8
    amsgrad = cfg.OPTIMIZER.AMSGRAD if cfg.OPTIMIZER.hasattr("AMSGRAD") else False
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr * cfg.OPTIMIZER.LR_FACTOR}, 
            {"params": model.decoder.parameters(), "lr": lr}, 
            # {"params": model.bottleneck.parameters(), "lr": lr}, 
        ], 
        lr=lr, 
        betas=betas, 
        eps=eps, 
        weight_decay=weight_decay, 
        amsgrad=amsgrad, 
    )
    return optimizer


def build_optimizer(cfg, model, *args, **kwargs):
    optimizer = _OPTIMIZER[cfg.OPTIMIZER.OPTIMIZER](cfg, model)
    # raise NotImplementedError("Function build_optimizer is not implemented.")
    return optimizer