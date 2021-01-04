
"""
Author  Yiqun Chen
Docs    Help build optimizer.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))



def build_optimizer(cfg, *args, **kwargs):
    raise NotImplementedError("Function build_optimizer is not implemented.")