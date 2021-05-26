
r"""
Info:
    Configurations for learning rate scheduler
Author:
    Yiqun Chen
"""

from alphaconfig import AlphaConfig
from .args import args

scheduler = AlphaConfig()

# ========      LinearLRScheduler       ========
scheduler.LinearLRScheduler.min_lr          =   2.5E-6
scheduler.LinearLRScheduler.warmup          =   10
# ========      StepLRScheduler         ========
scheduler.StepLRScheduler.min_lr            =   2.5E-6
scheduler.StepLRScheduler.warmup            =   10
scheduler.StepLRScheduler.update_epoch      =   range(int(args.max_epoch*0.1), args.max_epoch, int(args.max_epoch*0.1))
scheduler.StepLRScheduler.update_coeff      =   0.5
