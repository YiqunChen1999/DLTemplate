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
cfg.GENERAL.PIPLINE                             =   True
cfg.GENERAL.INFER_VERSION                       =   "0"

# ================================ 
# MODEL
# ================================ 
cfg.MODEL.DIR2CKPT                              =   os.path.join(cfg.GENERAL.ROOT, "checkpoints", cfg.GENERAL.ID)
cfg.MODEL.PATH2CKPT                             =   os.path.join(cfg.MODEL.DIR2CKPT, "{}.pth".format(cfg.GENERAL.ID))
cfg.MODEL.ENCODER.ARCH                          =   "" 
cfg.MODEL.DECODER.ARCH                          =   ""

# ================================ 
# DATA
# ================================ 
cfg.DATA.DIR                                    =   {
    "DATASET1": "/path/to/dataset1",
    "DATASET2": "/path/to/dataset2", 
}
cfg.DATA.NUM_WORKERS                            =   4
cfg.DATA.DATASETS                               =   ["DATASET1", "DATASET2"]
# ======== DATASET1 ========
cfg.DATA.DATASET1.TRAIN                         =   True
cfg.DATA.DATASET1.VALIE                         =   True
cfg.DATA.DATASET1.TEST                          =   True
cfg.DATA.DATASET1.INFER                         =   ["valid", "test"]
cfg.DATA.DATASET1.DATA_RATIO                    =   1.0
# ======== DATASET2 ========
cfg.DATA.DATASET2.TRAIN                         =   False
cfg.DATA.DATASET2.VALIE                         =   True
cfg.DATA.DATASET2.TEST                          =   True
cfg.DATA.DATASET2.INFER                         =   ["valid", "test"]
cfg.DATA.DATASET2.DATA_RATIO                    =   1.0

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
cfg.SCHEDULER.LinearLRScheduler.WARMUP_EPOCHS   =   0
# ========      StepLRScheduler         ========
cfg.SCHEDULER.StepLRScheduler.UPDATE_EPOCH      =   range(int(cfg.TRAIN.MAX_EPOCH*0.1), cfg.TRAIN.MAX_EPOCH, int(cfg.TRAIN.MAX_EPOCH*0.1))
cfg.SCHEDULER.StepLRScheduler.UPDATE_COEFF      =   0.5

# ================================ 
# LOSS_FN
# ================================ 
cfg.LOSS_FN.LOSS_FN                             =   "MSELoss" # ["MSELoss", "MAELoss"]
# ========      MSELoss     ========
cfg.LOSS_FN.MSELoss.WEIGHT1                     =   1.0
cfg.LOSS_FN.MSELoss.WEIGHT2                     =   1.0
# ========      MAELoss     ========
cfg.LOSS_FN.MAELoss.WEIGHT1                     =   1.0
cfg.LOSS_FN.MAELoss.WEIGHT2                     =   1.0

# ================================ 
# METRICS
# ================================ 
cfg.METRICS                                     =   ["ssim", "psnr"]


# ================================ 
# LOG
# ================================ 
cfg.SAVE.DIR                                    =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "results", cfg.GENERAL.ID, cfg.DATA.DATASET))
cfg.SAVE.SAVE                                   =   True

# ================================ 
# LOG
# ================================ 
cfg.LOG.DIR                                     =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "logs", cfg.GENERAL.ID))
   

# ================================ 
# CHECK
# ================================ 
cfg.cvt_state(read_only=True)

assert cfg.DATA.DATASET in cfg.DATA.DIR.keys(), "Unknown dataset {}".format(cfg.DATA.DATASET)

_paths = [
    cfg.LOG.DIR, 
    cfg.MODEL.DIR2CKPT, 
    cfg.SAVE.DIR, 
]
_paths.extend(list(cfg.DATA.DIR.cvt2dict().values()))

for _path in _paths:
    if not os.path.exists(_path):
        os.makedirs(_path)

