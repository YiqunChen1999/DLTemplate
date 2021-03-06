
r"""
Author:
    Yiqun Chen
Docs:
    Necessary modules for model.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils


class MyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MyModule, self).__init__()
        self._build()

    def _build(self):
        raise NotImplementedError("")

    def forward(self, inp):
        raise NotImplementedError("")


