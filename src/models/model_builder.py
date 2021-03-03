
r"""
Author:
    Yiqun Chen
Docs:
    Build model from configurations.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
from torch import nn
import torch.nn.functional as F

from utils import utils
from .text_encoder import _TEXT_ENCODER
from .video_encoder import _VIDEO_ENCODER
from .decoder import _DECODER


class RVOS(nn.Module):
    """
    The Language Guided Video Object Segmentation model
    """
    def __init__(self, cfg, split_size=2, *args, **kwargs):
        super(RVOS, self).__init__()
        self.cfg = cfg
        self.split_size = split_size
        self._build_model()
    
    def _build_model(self):
        # self.word_embedding = text_encoder_builder.WordEmbedding(self.cfg)
        self.decoder = _DECODER[self.cfg.MODEL.DECODER.ARCH](self.cfg)
        self.text_encoder = _TEXT_ENCODER[self.cfg.MODEL.TENCODER.ARCH](self.cfg)
        self.video_encoder = _VIDEO_ENCODER[self.cfg.MODEL.VENCODER.ARCH](self.cfg)

    def forward(self, text, frames, *args, **kwargs):
        # FIXME
        text = text.squeeze(1)
        if self.cfg.GENERAL.PIPLINE:
            # text = self.word_embedding(text, *args, **kwargs).squeeze(1)
            return self._pipline_model_forward(text, frames, *args, **kwargs)
        # text = self.word_embedding(text, *args, **kwargs).squeeze(1)
        # text = text.squeeze(1)
        text_repr = self.text_encoder(text, *args, **kwargs)
        video_repr = self.video_encoder(frames, text_repr, *args, **kwargs)
        outs = self.decoder(text_repr, video_repr, self.video_encoder, *args, **kwargs)
        return outs

    def _pipline_model_forward(self, text, frames, *args, **kwargs):
        device_0 = torch.device("cuda:{}".format(self.cfg.GENERAL.GPU[0]))
        device_1 = torch.device("cuda:{}".format(self.cfg.GENERAL.GPU[1]))
        
        text_splits = iter(text.split(self.split_size, dim=0))
        frames_splits = iter(frames.split(self.split_size, dim=0))
        text_next = next(text_splits)
        frames_next = next(frames_splits)
        
        text_repr_prev = self.text_encoder(text_next, *args, **kwargs)
        video_repr_prev = self.video_encoder(frames_next, text_repr_prev, *args, **kwargs)
        text_repr_prev = text_repr_prev.to(device_1)
        video_repr_prev = [v_repr.to(device_1) for v_repr in video_repr_prev]
        results = []
        
        for text_next, frames_next in zip(text_splits, frames_splits):
            outs = self.decoder(text_repr_prev, video_repr_prev, self.video_encoder, *args, **kwargs)
            results.append(outs)
            text_repr_prev =self.text_encoder(text_next, *args, **kwargs)
            video_repr_prev = self.video_encoder(frames_next, text_repr_prev, *args, **kwargs)
            text_repr_prev = text_repr_prev.to(device_1)
            video_repr_prev = [v_repr.to(device_1) for v_repr in video_repr_prev]
        outs = self.decoder(text_repr_prev, video_repr_prev, self.video_encoder, *args, **kwargs)
        results.append(outs)
        # outs = [
        #     torch.cat([o[0] for o in results]).to(device_0), 
        #     torch.cat([o[1] for o in results]).to(device_0), 
        #     torch.cat([o[2] for o in results]).to(device_0), 
        # ]
        outputs = {k: torch.cat([o[k] for o in results]).to(device_0) for k in results[0].keys()}

        return outputs


def build_model(cfg, logger=None):
    with utils.log_info(msg="Build model from configurations.", state=True, logger=logger):
        model = RVOS(cfg, split_size=2)
    return model