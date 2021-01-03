
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
args = parser.parse_args()

# ================================  GENERAL ================================
cfg.GENERAL.ID                                  =   "TemplateID{}".format(args.id)
cfg.GENERAL.STRICT_ID                           =   True if args.strict_id == "true" else False
cfg.GENERAL.BATCH_SIZE                          =   args.batch_size

# ================================  MODEL   ================================
cfg.MODEL.ARCH                                  =   None

# ================================  DATA    ================================
cfg.DATA.DIR                                    =   ""

# ================================  LOG     ================================
cfg.LOG.DIR                                     =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "logs", cfg.GENERAL.ID))



if cfg.GENERAL.STRICT_ID:
    assert not os.path.exists(cfg.LOG.DIR), "Cannot use same ID in strict mode."
    
_paths = [
    cfg.LOG.DIR
]

for _path in _paths:
    if not os.path.exists(_path):
        os.makedirs(_path)
