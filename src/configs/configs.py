
"""
Author  Yiqun Chen
Docs    Configurations, should not call other custom modules.
"""

import os, sys, copy, argparse
from attribdict import AttribDict as Dict

configs = Dict()
cfg = configs

parser = argparse.ArgumentParser()
parser.add_argument("id", type=str)
parser.add_argument("strict_id", default="true", choices=["true", "false"], type=str)
parser.add_argument("batch_size", type=int)
parser.add_argument("train", default="true", choices=["true", "false"], type=str)
parser.add_argument("eval", default="true", choices=["true", "false"], type=str)
parser.add_argument("test", default="false", choices=["true", "false"], type=str)
args = parser.parse_args()

# ================================ 
# GENERAL
# ================================ 
cfg.GENERAL.ID                                  =   "TemplateID{}".format(args.id)
cfg.GENERAL.STRICT_ID                           =   True if args.strict_id == "true" else False
cfg.GENERAL.BATCH_SIZE                          =   args.batch_size
cfg.GENERAL.TRAIN                               =   True if args.train == "true" else False
cfg.GENERAL.EVAL                                =   True if args.eval == "true" else False
cfg.GENERAL.TEST                                =   True if args.test == "true" else False

# ================================ 
# MODEL
# ================================ 
cfg.MODEL.ARCH                                  =   None
cfg.MODEL.ENCODER                               =   None
cfg.MODEL.DECODER                               =   None

# ================================ 
# DATA
# ================================ 
cfg.DATA.DIR                                    =   ""

# ================================ 
# OPTIMIZER
# ================================ 
cfg.OPTIMIZER.OPTIMIZER                         =   "Adam"
cfg.OPTIMIZER.LR                                =   1e-3

# ================================ 
# SCHEDULER
# ================================ 
cfg.SCHEDULER.SCHEDULER                         =   "Linear"

# ================================ 
# SCHEDULER
# ================================ 
cfg.TRAIN.MAX_EPOCH                             =   100

# ================================ 
# LOG
# ================================ 
cfg.LOG.DIR                                     =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "logs", cfg.GENERAL.ID))



if cfg.GENERAL.STRICT_ID:
    assert not os.path.exists(cfg.LOG.DIR), "Cannot use same ID in strict mode."
    
_paths = [
    cfg.LOG.DIR
]

for _path in _paths:
    if not os.path.exists(_path):
        os.makedirs(_path)
