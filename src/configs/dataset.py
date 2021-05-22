
r"""
Info:
    Configurations for datasets.
Author:
    Yiqun Chen
"""

from alphaconfig import AlphaConfig

data = AlphaConfig()

# ========      Dataset1    ========
data.Dataset1.dir                           =   "/home/chenyiqun/data/SOTS/outdoor"
data.Dataset1.train                         =   True
data.Dataset1.valid                         =   True
data.Dataset1.test                          =   False
data.Dataset1.infer                         =   ["valid"]
data.Dataset1.train_reso                    =   (256, 256)
data.Dataset1.extra_reso                    =   (128, 128)
data.Dataset1.data_ratio                    =   1.0

# ========      Dataset2  ========
data.Dataset2.dir                           =   "/home/chenyiqun/data/SOTS/indoor"
data.Dataset2.train                         =   False
data.Dataset2.valid                         =   True
data.Dataset2.test                          =   False
data.Dataset2.infer                         =   ["valid"]
data.Dataset2.train_reso                    =   (256, 256)
data.Dataset2.extra_reso                    =   (128, 128)
data.Dataset2.data_ratio                    =   1.0
