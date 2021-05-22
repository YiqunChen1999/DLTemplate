
r"""
Info:
    Configurations for optimizer
Author:
    Yiqun Chen
"""

from alphaconfig import AlphaConfig
from args import args

optim = AlphaConfig()

# ========      Adam        ========
optim.Adam.lr                           =   args.lr
optim.Adam.finetune                     =   1.0
optim.Adam.weight_decay                 =   0.0
# ========      AdamW       ========
optim.AdamW.lr                          =   args.lr
optim.AdamW.finetune                    =   1.0
optim.AdamW.weight_decay                =   0.1
