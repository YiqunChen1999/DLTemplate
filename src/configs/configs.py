r"""
Author:
    Yiqun Chen
Docs:
    Configurations, should not call other custom modules.
"""

import os, sys, copy
from alphaconfig import AlphaConfig

from .args import args
from .optim import optim
from .dataset import data
from .loss_fn import loss_fn
from .scheduler import scheduler

configs = AlphaConfig()
cfg = configs

# ================================ 
# GENERAL
# ================================ 
cfg.gnrl.root                                   =   os.path.join(os.getcwd(), ".")
cfg.gnrl.id                                     =   "{}".format(args.id)
cfg.gnrl.batch                                  =   args.batch_size
cfg.gnrl.resume                                 =   True if args.resume == "true" else False
cfg.gnrl.cuda                                   =   eval(args.cuda)
cfg.gnrl.ckphs                                  =   range(int(args.max_epoch*0.6), args.max_epoch, int(args.max_epoch*0.1))
cfg.gnrl.infer                                  =   "0"

# ================================ 
# MODEL
# ================================ 
cfg.model.ckpts                                 =   os.path.join(cfg.gnrl.root, "checkpoints", cfg.gnrl.id)
cfg.model.path2ckpt                             =   os.path.join(cfg.model.ckpts, "{}.pth".format(cfg.gnrl.id))
cfg.model.enc.arch                              =   "ResNeSt101EncoderV0" 
cfg.model.dec.arch                              =   "HighResoDecoderV0"

# ================================ 
# DATA
# ================================ 
cfg.data.aug                                    =   True
cfg.data.mean                                   =   [0., 0., 0.]
cfg.data.norm                                   =   [255., 255., 255.]
cfg.data.num_workers                            =   8
cfg.data.datasets                               =   ["Dataset1", "Dataset1"]

# ================================ 
# OPTIMIZER
# ================================ 
cfg.optim.optim                                 =   "Adam" 

# ================================ 
# TRAIN
# ================================ 
cfg.train.max_epoch                             =   args.max_epoch 

# ================================ 
# SCHEDULER
# ================================ 
cfg.scheduler.scheduler                         =   "LinearLRScheduler" # ["LinearLRScheduler", "StepLRScheduler"]

# ================================ 
# LOSS_FN
# ================================ 
cfg.loss_fn.loss_fn                             =   "MSESSIMLoss" # ["MSELoss", "MAELoss"]

# ================================ 
# METRICS
# ================================ 
cfg.metrics                                     =   ["ssim", "psnr"]

# ================================ 
# LOG
# ================================ 
cfg.save.dir                                    =   os.path.join(os.path.join(cfg.gnrl.root, "results", cfg.gnrl.id))
cfg.save.save                                   =   True

# ================================ 
# LOG
# ================================ 
cfg.log.dir                                     =   os.path.join(os.path.join(cfg.gnrl.root, "logs", cfg.gnrl.id))

# ================================ 
# Other Configurations
# ================================ 
cfg.data[cfg.data.data]                         =   data[cfg.data.data]
cfg.optim[cfg.optim.optim]                      =   optim[cfg.optim.optim]
cfg.loss_fn[cfg.loss_fn.loss_fn]                =   loss_fn[cfg.loss_fn.loss_fn]
cfg.scheduler[cfg.scheduler.scheduler]          =   scheduler[cfg.scheduler.scheduler]

cfg.cvt_state(read_only=True)



