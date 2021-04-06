r"""
Author:
    Yiqun Chen
Docs:
    Configurations, should not call other custom modules.
"""

import os, sys, copy, argparse
from alphaconfig import AlphaConfig

configs = AlphaConfig()
cfg = configs

parser = argparse.ArgumentParser()
parser.add_argument("--id",                             type=str,   required=True)
parser.add_argument("--batch_size",                     type=int,   required=True)
parser.add_argument("--lr",         default=1e-4,       type=float, required=True)
parser.add_argument("--max_epoch",  default=20,         type=int,   required=True)
parser.add_argument("--resume",     default="false",    type=str,   required=True,  choices=["true", "false"])
parser.add_argument("--gpu",                            type=str,   required=True)
args = parser.parse_args()

# ================================ 
# GENERAL
# ================================ 
cfg.GENERAL.ROOT                                =   os.path.join(os.getcwd(), ".")
cfg.GENERAL.ID                                  =   "{}".format(args.id)
cfg.GENERAL.BATCH_SIZE                          =   args.batch_size
cfg.GENERAL.RESUME                              =   True if args.resume == "true" else False
cfg.GENERAL.GPU                                 =   eval(args.gpu)
cfg.GENERAL.CHECK_EPOCHS                        =   range(int(args.max_epoch*0.6), args.max_epoch, int(args.max_epoch*0.1))
cfg.GENERAL.PIPLINE                             =   False
cfg.GENERAL.INFER_VERSION                       =   "0"

# ================================ 
# MODEL
# ================================ 
cfg.MODEL.DIR2CKPT                              =   os.path.join(cfg.GENERAL.ROOT, "checkpoints", cfg.GENERAL.ID)
cfg.MODEL.PATH2CKPT                             =   os.path.join(cfg.MODEL.DIR2CKPT, "{}.pth".format(cfg.GENERAL.ID))
cfg.MODEL.ENCODER.ARCH                          =   "ResNeSt101EncoderV0" 
cfg.MODEL.DECODER.ARCH                          =   "HighResoDecoderV0"

# ================================ 
# DATA
# ================================ 
cfg.DATA.AUGMENTATION                           =   True
cfg.DATA.MEAN                                   =   [0., 0., 0.]
cfg.DATA.NORM                                   =   [255., 255., 255.]
cfg.DATA.NUM_WORKERS                            =   64
cfg.DATA.DATASETS                               =   ["Dataset1", "Dataset1"]
# ========      Dataset1    ========
cfg.DATA.Dataset1.DIR                           =   "/home/chenyiqun/data/SOTS/outdoor"
cfg.DATA.Dataset1.TRAIN                         =   True
cfg.DATA.Dataset1.VALID                         =   True
cfg.DATA.Dataset1.TEST                          =   False
cfg.DATA.Dataset1.INFER                         =   ["valid"]
cfg.DATA.Dataset1.TRAIN_RESO                    =   (256, 256)
cfg.DATA.Dataset1.EXTRA_RESO                    =   (128, 128)
cfg.DATA.Dataset1.DATA_RATIO                    =   1.0
# ========      Dataset2  ========
cfg.DATA.Dataset2.DIR                           =   "/home/chenyiqun/data/SOTS/indoor"
cfg.DATA.Dataset2.TRAIN                         =   False
cfg.DATA.Dataset2.VALID                         =   True
cfg.DATA.Dataset2.TEST                          =   False
cfg.DATA.Dataset2.INFER                         =   ["valid"]
cfg.DATA.Dataset2.TRAIN_RESO                    =   (256, 256)
cfg.DATA.Dataset2.EXTRA_RESO                    =   (128, 128)
cfg.DATA.Dataset2.DATA_RATIO                    =   1.0

# ================================ 
# OPTIMIZER
# ================================ 
cfg.OPTIMIZER.OPTIMIZER                         =   "Adam" 
# ========      Adam        ========
cfg.OPTIMIZER.Adam.LR                           =   args.lr
cfg.OPTIMIZER.Adam.FINETUNE                     =   1.0
cfg.OPTIMIZER.Adam.WEIGHT_DECAY                 =   0.0
# ========      AdamW       ========
cfg.OPTIMIZER.AdamW.LR                          =   args.lr
cfg.OPTIMIZER.AdamW.FINETUNE                    =   1.0
cfg.OPTIMIZER.AdamW.WEIGHT_DECAY                =   0.1

# ================================ 
# TRAIN
# ================================ 
cfg.TRAIN.MAX_EPOCH                             =   args.max_epoch 

# ================================ 
# SCHEDULER
# ================================ 
cfg.SCHEDULER.SCHEDULER                         =   "LinearLRScheduler" # ["LinearLRScheduler", "StepLRScheduler"]
# ========      LinearLRScheduler       ========
cfg.SCHEDULER.LinearLRScheduler.MIN_LR          =   2.5E-6
cfg.SCHEDULER.LinearLRScheduler.WARMUP_EPOCHS   =   10
# ========      StepLRScheduler         ========
cfg.SCHEDULER.StepLRScheduler.MIN_LR            =   2.5E-6
cfg.SCHEDULER.StepLRScheduler.WARMUP_EPOCHS     =   10
cfg.SCHEDULER.StepLRScheduler.UPDATE_EPOCH      =   range(int(cfg.TRAIN.MAX_EPOCH*0.1), cfg.TRAIN.MAX_EPOCH, int(cfg.TRAIN.MAX_EPOCH*0.1))
cfg.SCHEDULER.StepLRScheduler.UPDATE_COEFF      =   0.5

# ================================ 
# LOSS_FN
# ================================ 
cfg.LOSS_FN.LOSS_FN                             =   "MSESSIMLoss" # ["MSELoss", "MAELoss"]
# ========      MSELoss         ========
cfg.LOSS_FN.MSELoss.WEIGHT1                     =   1.0
cfg.LOSS_FN.MSELoss.WEIGHT2                     =   1.0
# ========      MAELoss         ========
cfg.LOSS_FN.MAELoss.WEIGHT1                     =   1.0
cfg.LOSS_FN.MAELoss.WEIGHT2                     =   1.0
# ========      MSESSIMLoss     ========
cfg.LOSS_FN.MSESSIMLoss.MSE                     =   1.0
cfg.LOSS_FN.MSESSIMLoss.SSIM                    =   0.5

# ================================ 
# METRICS
# ================================ 
cfg.METRICS                                     =   ["ssim", "psnr"]


# ================================ 
# LOG
# ================================ 
cfg.SAVE.DIR                                    =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "results", cfg.GENERAL.ID))
cfg.SAVE.SAVE                                   =   True

# ================================ 
# LOG
# ================================ 
cfg.LOG.DIR                                     =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "logs", cfg.GENERAL.ID))
   

# ================================ 
# CHECK
# ================================ 
cfg.cvt_state(read_only=True)



