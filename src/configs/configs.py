r"""
Author:
    Yiqun Chen
Docs:
    Configurations, should not call other custom modules.
"""

import os, sys, copy, argparse
from attribdict import AttribDict as Dict

configs = Dict()
cfg = configs

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_size", type=float, default=1.0)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--lr", type=float, default=2e-5 * 8)
parser.add_argument("--max_epoch", type=int, default=400)
parser.add_argument("--resume", default="false", choices=["true", "false"], type=str, required=True)
parser.add_argument("--train", default="true", choices=["true", "false"], type=str, required=True)
parser.add_argument("--valid", default="true", choices=["true", "false"], type=str, required=True)
parser.add_argument("--test", default="false", choices=["true", "false"], type=str, required=True)
parser.add_argument("--gpu", type=str, required=True)
args = parser.parse_args()

# ================================ 
# GENERAL
# ================================ 
cfg.GENERAL.ROOT                                =   os.path.join(os.getcwd(), ".")
cfg.GENERAL.ID                                  =   "{}".format(args.id)
cfg.GENERAL.BATCH_SIZE                          =   args.batch_size
cfg.GENERAL.RESUME                              =   True if args.resume == "true" else False
cfg.GENERAL.TRAIN                               =   True if args.train == "true" else False
cfg.GENERAL.VALID                               =   True if args.valid == "true" else False
cfg.GENERAL.TEST                                =   True if args.test == "true" else False
cfg.GENERAL.GPU                                 =   eval(args.gpu)

# ================================ 
# MODEL
# ================================ 
cfg.MODEL.ARCH                                  =   None # TODO
cfg.MODEL.ENCODER                               =   None
cfg.MODEL.DECODER                               =   None
cfg.MODEL.CKPT_DIR                              =   os.path.join(cfg.GENERAL.ROOT, "checkpoints", cfg.GENERAL.ID)
cfg.MODEL.PATH2CKPT                             =   os.path.join(cfg.MODEL.CKPT_DIR, "{}.pth".format(cfg.GENERAL.ID))

# ================================ 
# DATA
# ================================ 
cfg.DATA.DIR                                    =   {
    "Dataset": "path2dataset", 
}
cfg.DATA.NUMWORKERS                             =   4
cfg.DATA.DATASET                                =   args.dataset # "MNHHAZE"
cfg.DATA.MEAN                                   =   [0., 0., 0.]
cfg.DATA.NORM                                   =   [255, 255, 255]
cfg.DATA.AUGMENTATION                           =   True

# ================================ 
# OPTIMIZER
# ================================ 
cfg.OPTIMIZER.OPTIMIZER                         =   "AdamW" 
cfg.OPTIMIZER.LR                                =   args.lr 
cfg.OPTIMIZER.LR_FACTOR                         =   1.0

# ================================ 
# SCHEDULER
# ================================ 
cfg.TRAIN.MAX_EPOCH                             =   args.max_epoch 
cfg.TRAIN.RANDOM_SAMPLE_RATIO                   =   args.data_size 

# ================================ 
# SCHEDULER
# ================================ 
cfg.SCHEDULER.SCHEDULER                         =   "StepLRScheduler"
cfg.SCHEDULER.UPDATE_EPOCH                      =   range(int(cfg.TRAIN.MAX_EPOCH*0.1), cfg.TRAIN.MAX_EPOCH, int(cfg.TRAIN.MAX_EPOCH*0.1))
cfg.SCHEDULER.UPDATE_COEFF                      =   0.5

# ================================ 
# LOSS_FN
# ================================ 
cfg.LOSS_FN.LOSS_FN                             =   "MSELoss" 
cfg.LOSS_FN.WEIGHTS                             =   {
    "Item": "Weight(float)"
}

# ================================ 
# LOG
# ================================ 
cfg.SAVE.DIR                                    =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "results", cfg.GENERAL.ID, cfg.DATA.DATASET))
cfg.SAVE.SAVE                                   =   True

# ================================ 
# LOG
# ================================ 
cfg.LOG.DIR                                     =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "logs", cfg.GENERAL.ID))
   

assert cfg.DATA.DATASET in cfg.DATA.DIR.keys(), "Unknown dataset {}".format(cfg.DATA.DATASET)

_paths = [
    cfg.LOG.DIR, 
    cfg.MODEL.CKPT_DIR, 
    cfg.SAVE.DIR, 
]
_paths.extend(list(cfg.DATA.DIR.as_dict().values()))

for _path in _paths:
    if not os.path.exists(_path):
        os.makedirs(_path)

