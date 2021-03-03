
r"""
Author:
    Yiqun Chen
Docs:
    Encoder classes.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

_TEXT_ENCODER = {}

def add_encoder(encoder):
    _TEXT_ENCODER[encoder.__name__] = encoder
    return encoder


@add_encoder
class ConvTEncoderV1(nn.Module):
    """"""
    def __init__(self, cfg, *args, **kwargs):
        super(ConvTEncoderV1, self).__init__()
        self.cfg = cfg
        self.in_dim = self.cfg.DATA.QUERY.BERT_DIM
        self.out_dim = self.cfg.DATA.QUERY.DIM
        self.max_words = self.cfg.DATA.QUERY.MAX_WORDS
        self._build_model()

    def _build_model(self):
        self.conv = nn.Conv1d(self.in_dim, self.out_dim, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, text_repr, *args, **kwargs):
        text_repr = text_repr.mean(dim=1)
        assert len(text_repr.shape) == 3, "Dimension Error"
        text_repr = text_repr.permute(0, 2, 1)
        text_repr = self.tanh(self.conv(text_repr))
        text_repr = text_repr.permute(0, 2, 1)
        return text_repr


if __name__ == "__main__":
    print(_ENCODER)