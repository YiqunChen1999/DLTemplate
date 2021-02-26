
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
from .modules import *

_ENCODER = {}

def add_encoder(encoder):
    _ENCODER[encoder.__name__] = encoder
    return encoder


@add_encoder
class ResInterI3DVEncoderV1(nn.Module):
    """"""
    def __init__(self, cfg, *args, **kwargs):
        super(ResInterI3DVEncoderV1, self).__init__()
        self.cfg = cfg
        self.num_classes = self.cfg.DATA.VIDEO.CLASS_NUM
        self.pretrained = self.cfg.MODEL.VIDEO_ENCODER.PRETRAINED
        self.pretrained_path = self.cfg.MODEL.VIDEO_ENCODER.CHECKPOINT
        self.args = args
        self.kwargs = kwargs
        self._build_model()
        self._load_pretrained_parameters()

    def _build_model(self):
        self.model = _Inflated3DConvNet()

        # self.fusion_1 = EarlyFusion(300, 64)
        # self.fusion_2 = EarlyFusion(300, 192)
        # self.fusion_3 = EarlyFusion(300, 480)
        # self.fusion_4 = EarlyFusion(300, 832)
        
        # self.fusion_1 = EarlyFusionWithCBN(300, 64)
        # self.fusion_2 = EarlyFusionWithCBN(300, 192)
        # self.fusion_3 = EarlyFusionWithCBN(300, 480)
        # self.fusion_4 = EarlyFusionWithCBN(300, 832)
        
        # self.fusion_1 = EarlyFusionWithFiLM(300, 64)
        # self.fusion_2 = EarlyFusionWithFiLM(300, 192)
        # self.fusion_3 = EarlyFusionWithFiLM(300, 480)
        # self.fusion_4 = EarlyFusionWithFiLM(300, 832)

    def _load_pretrained_parameters(self):
        if self.pretrained:
            assert self.pretrained_path is not None, \
                "the path to pretrained parameters is none, please check it."
            print(">>>> loading pretrained backbone parameters...")
            self.model.load_state_dict(
                torch.load(self.pretrained_path, map_location=lambda storage, loc: storage)
            )
            # NOTE Here we train the whole video encoding
            for param in self.model.conv3d_1a_7x7.parameters():
                param.requires_grad = False
            for param in self.model.conv3d_2b_1x1.parameters():
                param.requires_grad = False
            # for param in self.model.parameters():
            #     param.requires_grad = False
            # for param in self.model.maxPool3d_4a_3x3.parameters():
            #     param.requires_grad = True
            # for param in self.model.mixed_4b.parameters():
            #     param.requires_grad = True
            # for param in self.model.mixed_4c.parameters():
            #     param.requires_grad = True
            # for param in self.model.mixed_4d.parameters():
            #     param.requires_grad = True
            # for param in self.model.mixed_4e.parameters():
            #     param.requires_grad = True
            # for param in self.model.mixed_4f.parameters():
            #     param.requires_grad = True
        if not self.cfg.GENERAL.TRAIN:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, frames, text_repr, *args, **kwargs):
        frames = frames.transpose(-3, -4)
        video_repr = []

        # Preprocessing
        # NOTE out.shape: [N, 64, 8, 256, 256] when frames.shape: [N, 3, 16, 512, 512]
        out = self.model.conv3d_1a_7x7(frames)
        # out = self.fusion_1(text_repr, out)
        # video_repr.append(out)

        # out.shape: [N, 64, 8, 128, 128] when frames.shape: [N, 3, 16, 512, 512]
        out = self.model.maxPool3d_2a_3x3(out)
        # out.shape: [N, 64, 8, 128, 128] when frames.shape: [N, 3, 16, 512, 512]
        out = self.model.conv3d_2b_1x1(out)
        # NOTE out.shape: [N, 192, 8, 128, 128] when frames.shape: [N, 3, 16, 512, 512]
        out = self.model.conv3d_2c_3x3(out)
        out = self.fusion_2(text_repr, out)
        video_repr.append(out)

        # out.shape: [N, 192, 8, 64, 64] when frames.shape: [N, 3, 16, 512, 512]
        out = self.model.maxPool3d_3a_3x3(out)
        # out.shape: [N, 256, 8, 64, 64] when frames.shape: [N, 3, 16, 512, 512]
        out = self.model.mixed_3b(out)
        # out = self.SEModules["module_1"](out)
        # NOTE out.shape: [N, 480, 8, 64, 64] when frames.shape: [N, 3, 16, 512, 512]
        out = self.model.mixed_3c(out)
        # out = self.SEModules["module_2"](out)
        out = self.fusion_3(text_repr, out)
        # video_repr.append(out)

        # out.shape: [N, 480, 4, 32, 32] when frames.shape: [N, 3, 16, 512, 512]
        out = self.model.maxPool3d_4a_3x3(out)
        # out.shape: [N, 512, 4, 32, 32] when frames.shape: [N, 3, 16, 512, 512]
        out = self.model.mixed_4b(out)
        # out = self.SEModules["module_3"](out)
        # out.shape: [N, 512, 4, 32, 32] when frames.shape: [N, 3, 16, 512, 512]
        out = self.model.mixed_4c(out)
        # out = self.SEModules["module_4"](out)
        # out.shape: [N, 512, 4, 32, 32] when frames.shape: [N, 3, 16, 512, 512]
        out = self.model.mixed_4d(out)
        # out = self.SEModules["module_5"](out)
        # out.shape: [N, 528, 4, 32, 32] when frames.shape: [N, 3, 16, 512, 512]
        out = self.model.mixed_4e(out)
        # out = self.SEModules["module_6"](out)
        # NOTE out.shape: [N, 832, 4, 32, 32] when frames.shape: [N, 3, 16, 512, 512]
        out = self.model.mixed_4f(out)
        # out = self.SEModules["module_7"](out)
        out = self.fusion_4(text_repr, out)
        video_repr.append(out)

        return video_repr




if __name__ == "__main__":
    print(_ENCODER)