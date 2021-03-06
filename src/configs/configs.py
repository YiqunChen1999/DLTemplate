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
parser.add_argument("--data_size", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--max_epoch", type=int, default=20, required=True)
parser.add_argument("--resume", default="false", choices=["true", "false"], type=str, required=True)
parser.add_argument("--train", default="true", choices=["true", "false"], type=str, required=True)
parser.add_argument("--valid", default="true", choices=["true", "false"], type=str, required=True)
parser.add_argument("--test", default="false", choices=["true", "false"], type=str, required=True)
parser.add_argument("--infer", default="none", choices=["train", "valid", "test", "none"], type=str)
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
cfg.GENERAL.INFER                               =   args.infer
cfg.GENERAL.GPU                                 =   eval(args.gpu)
cfg.GENERAL.PIPLINE                             =   True

# ================================ 
# MODEL
# ================================ 
cfg.MODEL.CKPT_DIR                              =   os.path.join(cfg.GENERAL.ROOT, "checkpoints", cfg.GENERAL.ID)
cfg.MODEL.PATH2CKPT                             =   os.path.join(cfg.MODEL.CKPT_DIR, "{}.pth".format(cfg.GENERAL.ID))
cfg.MODEL.TENCODER.ARCH                         =   "ConvTEncoderV1" # TODO
cfg.MODEL.VENCODER.ARCH                         =   "ResInterI3DVEncoderV1"
cfg.MODEL.VENCODER.PRETRAINED                   =   True
cfg.MODEL.VENCODER.CHECKPOINT                   =   os.path.join(configs.GENERAL.ROOT, "checkpoints", "i3d_rgb.pth")
cfg.MODEL.DECODER.ARCH                          =   "MSMFCNDecoderV1"

# ================================ 
# DATA
# ================================ 
cfg.DATA.DIR                                    =   {
    "YRVOS2021FPS6": "/data/YoutubeRVOS2021", 
    "YoutubeRVOS2021": "/data/YoutubeRVOS2021", 
    "YoutubeRVOS2021FPS6": "/data/YoutubeRVOS2021", 
    "A2DSentences": "/data/A2D/Develop/raw", 
    "JHMDBSentences": "/data/JHMDB/raw", 
}
cfg.DATA.NUMWORKERS                             =   4
cfg.DATA.DATASET                                =   args.dataset 
cfg.DATA.RANDOM_SAMPLE_RATIO                    =   args.data_size
cfg.DATA.MULTI_FRAMES                           =   False
cfg.DATA.VIDEO.MEAN                             =   [0., 0., 0.]
cfg.DATA.VIDEO.NORM                             =   [255, 255, 255]
cfg.DATA.VIDEO.FORMAT                           =   "RGB"
cfg.DATA.VIDEO.NUM_FRAMES                       =   16
cfg.DATA.VIDEO.RESOLUTION                       =   ((512, 512), (128, 128), (32, 32)) # ((256, 256), (64, 64), (16, 16))
cfg.DATA.VIDEO.REPR_CHANNELS                    =   (832, 256, 128)
cfg.DATA.VIDEO.SPATIAL_FEAT_DIM                 =   8
cfg.DATA.QUERY.MAX_WORDS                        =   20
cfg.DATA.QUERY.BERT_DIM                         =   768
cfg.DATA.QUERY.DIM                              =   300

# ================================ 
# OPTIMIZER
# ================================ 
cfg.OPTIMIZER.OPTIMIZER                         =   "Adam" 
cfg.OPTIMIZER.LR                                =   args.lr 
cfg.OPTIMIZER.FINETUNE_FACTOR                   =   1.0

# ================================ 
# SCHEDULER
# ================================ 
cfg.TRAIN.MAX_EPOCH                             =   args.max_epoch 
cfg.TRAIN.RANDOM_SAMPLE_RATIO                   =   args.data_size 

# ================================ 
# SCHEDULER
# ================================ 
cfg.SCHEDULER.SCHEDULER                         =   "LinearLRScheduler" # ["LinearLRScheduler", "StepLRScheduler"]
cfg.SCHEDULER.UPDATE_EPOCH                      =   range(int(cfg.TRAIN.MAX_EPOCH*0.1), cfg.TRAIN.MAX_EPOCH, int(cfg.TRAIN.MAX_EPOCH*0.1))
cfg.SCHEDULER.UPDATE_COEFF                      =   0.5
cfg.SCHEDULER.MIN_LR                            =   2.5e-6
cfg.SCHEDULER.WARMUP_EPOCHS                     =   0

# ================================ 
# LOSS_FN
# ================================ 
cfg.LOSS_FN.LOSS_FN                             =   "MaskBCEBBoxMSELoss" 
cfg.LOSS_FN.WEIGHTS                             =   {
    "POS_WEIGHT": 1.5, "BBOX_COEFF": 1.5, 
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

