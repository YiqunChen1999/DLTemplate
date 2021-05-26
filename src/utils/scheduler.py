
r"""
Author:
    Yiqun Chen
Docs:
    Help build lr scheduler.
"""

import os, sys, warnings
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from .logger import logger

SCHEDULER = {}

def add_scheduler(scheduler):
    SCHEDULER[scheduler.__name__] = scheduler
    return scheduler


@add_scheduler
class StepLRScheduler:
    def __init__(self, cfg, optimizer: torch.optim.Optimizer, *args, **kwargs):
        super(StepLRScheduler, self).__init__()
        self.cfg = cfg
        self.optimizer = optimizer
        self._build_()

    def _build_(self):
        self.warmup = self.cfg.scheduler.StepLRScheduler.warmup if hasattr(self.cfg.scheduler.StepLRScheduler, "WARMUP_EPOCHS") else 0
        self.update_epoch = list(self.cfg.scheduler.StepLRScheduler.update_epoch)
        self.update_coeff = self.cfg.scheduler.StepLRScheduler.update_coeff
        self.epoch = 1

    def update(self):
        old_lrs = []
        new_lrs = []
        if self.epoch in self.update_epoch:
            for param_group in self.optimizer.param_groups:
                old_lrs.append(param_group["lr"])
                new_lrs.append(old_lrs[-1]*self.update_coeff)
                assert len(old_lrs) == len(new_lrs)
                if new_lrs[-1] <= 0:
                    warnings.warn("Learning rate {} is not larger than 0.0.".format(new_lrs[-1]))
                param_group["lr"] = new_lrs[-1]
        self.epoch += 1

    def step(self):
        self.update()

    def state_dict(self):
        state_dict = {
            "cfg": self.cfg, 
            "cnt": self.epoch, 
            "warmup_epochs": self.warmup, 
            "update_epoch": self.update_epoch, 
            "update_coeff": self.update_coeff, 
        }
        return state_dict

    def load_state_dict(self, state_dict: dict):
        self.cfg = state_dict.pop("cfg", self.cfg)
        self.epoch = state_dict.pop("cnt", self.epoch)
        self.warmup = state_dict.pop("warmup_epochs", self.warmup)
        self.update_coeff = state_dict.pop("update_coeff", self.update_coeff)
        self.update_epoch = state_dict.pop("update_epoch", self.update_epoch)
        
    def sychronize(self, epoch):
        self.epoch = epoch


@add_scheduler
class LinearLRScheduler:
    def __init__(self, cfg, optimizer, *args, **kwargs):
        super(LinearLRScheduler, self).__init__()
        self.cfg = cfg
        self.optimizer = optimizer
        self._build_()

    def _build_(self):
        self.min_lr = self.cfg.scheduler.LinearLRScheduler.min_lr
        self.warmup = self.cfg.scheduler.LinearLRScheduler.warmup if hasattr(self.cfg.scheduler.LinearLRScheduler, "WARMUP_EPOCHS") else 0
        self.max_epoch = self.cfg.train.max_epoch - self.warmup
        self.lr_info = [{"init_lr": param_group["lr"], "final_lr": self.min_lr} for param_group in self.optimizer.param_groups]
        if isinstance(self.min_lr, list):
            for idx in range(len(self.lr_info)):
                self.lr_info[idx]["final_lr"] = self.min_lr[idx]
        for idx in range(len(self.lr_info)):
            self.lr_info[idx]["delta_lr"] = (self.lr_info[idx]["init_lr"] - self.lr_info[idx]["final_lr"]) / self.max_epoch
        self.epoch = 1
        if self.warmup > 0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 1e-8

    def update(self):
        for idx, param_group in enumerate(self.optimizer.param_groups):
            if self.epoch < self.warmup:
                param_group["lr"] = round(self.lr_info[idx]["init_lr"]*self.epoch/self.warmup, 9)
            else:
                param_group["lr"] = round(self.lr_info[idx]["init_lr"] - (self.epoch-self.warmup) * self.lr_info[idx]["delta_lr"], 9)
            if param_group["lr"] < 0:
                logger.raise_error(ValueError, "Expect positive learning rate but got {}".format(param_group["lr"]))
        self.epoch += 1

    def step(self):
        self.update()

    def state_dict(self):
        state_dict = {
            "cfg": self.cfg, 
            "cnt": self.epoch, 
            "min_lr": self.min_lr, 
            "max_epoch": self.max_epoch, 
            "lr_info": self.lr_info,
            "warmup_epochs": self.warmup, 
        }
        return state_dict

    def load_state_dict(self, state_dict: dict):
        self.cfg = state_dict.pop("cfg", self.cfg)
        self.epoch = state_dict.pop("cnt", self.epoch)
        self.min_lr = state_dict.pop("min_lr", self.min_lr)
        self.max_epoch = state_dict.pop("max_epoch", self.max_epoch)
        self.lr_info = state_dict.pop("lr_info", self.lr_info)
        self.warmup = state_dict.pop("warmup_epochs", self.warmup)

    def sychronize(self, epoch):
        self.epoch = epoch


def build_scheduler(cfg, optimizer, logger=None, *args, **kwargs):
    with logger.log_info(msg="Build learning rate scheduler", level="INFO", state=True, logger=logger):
        return SCHEDULER[cfg.scheduler.scheduler](cfg, optimizer, *args, **kwargs)