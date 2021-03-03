
r"""
Author:
    Yiqun Chen
Docs:
    Decoder classes.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils
from .modules import (
    AsymmCrossAttnV1, 
    AsymmCrossAttnV2, 
    MFCNModuleV2, 
)

_DECODER = {}

def add_decoder(decoder):
    _DECODER[decoder.__name__] = decoder
    return decoder


@add_decoder
class MSMFCNDecoderV2(torch.nn.Module):
    """
    This module can deal with multi-task.
    """
    def __init__(self, cfg, *args, **kwargs):
        super(MSMFCNDecoderV2, self).__init__()
        self.cfg = cfg
        self.args = args
        self.kwargs = kwargs
        self.text_repr_dim = self.cfg.DATA.QUERY.DIM
        self.video_repr_channels = self.cfg.DATA.VIDEO.REPR_CHANNELS
        self.spatial_feat_dim = self.cfg.DATA.VIDEO.SPATIAL_FEAT_DIM
        self.resolution = self.cfg.DATA.VIDEO.RESOLUTION
        self.num_frames = self.cfg.DATA.VIDEO.NUM_FRAMES
        self._build()

    def _build(self):
        # Linear transformation from video to text.
        self.asymm_cross_attn = AsymmCrossAttnV2(self.text_repr_dim, self.video_repr_channels, self.spatial_feat_dim, self.resolution)
        
        self.deconv_vs2m = nn.Sequential(
            nn.ConvTranspose2d(
                self.video_repr_channels[0], self.video_repr_channels[1], kernel_size=8, stride=4, padding=2, bias=True
            ), 
            nn.Conv2d(
                self.video_repr_channels[1], self.video_repr_channels[1], kernel_size=3, stride=1, padding=1, bias=True
            ),
        )
        self.deconv_vm2l = nn.Sequential(
            nn.ConvTranspose2d(
                self.video_repr_channels[1], self.video_repr_channels[2], kernel_size=8, stride=4, padding=2, bias=True
            ), 
            nn.Conv2d(
                self.video_repr_channels[2], self.video_repr_channels[2], kernel_size=3, stride=1, padding=1, bias=True
            ),
        )
        self.deconv_v = nn.Sequential(
            nn.ConvTranspose3d(
                self.video_repr_channels[0], self.video_repr_channels[0], kernel_size=(8, 1, 1), stride=(4, 1, 1), padding=(2, 0, 0), bias=True
            ), 
            nn.Conv3d(
                self.video_repr_channels[0], self.video_repr_channels[0], kernel_size=3, stride=1, padding=1, bias=True
            ),
        )
        # self.deconv_t = nn.Sequential(
        #     nn.ConvTranspose3d(
        #         self.text_repr_dim, self.text_repr_dim, kernel_size=(8, 1, 1), stride=(4, 1, 1), padding=(2, 0, 0), bias=True
        #     ), 
        #     nn.Conv3d(
        #         self.text_repr_dim, self.text_repr_dim, kernel_size=3, stride=1, padding=1, bias=True
        #     ),
        # )

        self.v_gmax = nn.AdaptiveMaxPool2d(1)
        
        self.fcn_s = MFCNModuleV2(self.text_repr_dim, 2*self.text_repr_dim+self.video_repr_channels[0], self.video_repr_channels[0], self.video_repr_channels[1], self.spatial_feat_dim)
        self.fcn_m = MFCNModuleV2(self.text_repr_dim, 2*self.text_repr_dim+self.video_repr_channels[0], self.video_repr_channels[1], self.video_repr_channels[2], self.spatial_feat_dim)
        self.fcn_l = MFCNModuleV2(self.text_repr_dim, 2*self.text_repr_dim+self.video_repr_channels[0], self.video_repr_channels[2], self.video_repr_channels[2]//2, self.spatial_feat_dim)
        
        self.fcn_fusion = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.text_max_pool = nn.AdaptiveMaxPool1d(1)
        self.F_va_max_pool = nn.AdaptiveMaxPool2d(1, 1)
        self.linear_vm = nn.Conv2d(192, self.video_repr_channels[1], kernel_size=1)

    def forward(self, text_repr, video_repr, *args, **kwargs):
        spatial_maps = kwargs["spatial_feats"]
        spatial_map_s, spatial_map_m, spatial_map_l = spatial_maps
        batch_size, chann_s, temper_s, hs, ws = video_repr[-1].shape
        batch_size, chann_m, temper_m, hm, wm = video_repr[-2].shape
        spatial_maps_s = spatial_map_s.repeat(batch_size*temper_s, 1, 1, 1)
        spatial_maps_m = spatial_map_m.repeat(batch_size*self.cfg.DATA.VIDEO.NUM_FRAMES, 1, 1, 1)
        spatial_maps_l = spatial_map_l.repeat(batch_size*self.cfg.DATA.VIDEO.NUM_FRAMES, 1, 1, 1)
        text_repr = text_repr.repeat(temper_s, 1, 1)
        F_ta, F_va = self.asymm_cross_attn(text_repr, video_repr[-1], spatial_maps_s)

        F_va = self.deconv_v(F_va.reshape(batch_size, -1, temper_s, hs, ws))
        F_va = F_va.reshape(batch_size*self.cfg.DATA.VIDEO.NUM_FRAMES, -1, hs, ws)
        # F_ta = self.deconv_t(F_ta.reshape(batch_size, -1, temper_s, hs, ws)).reshape(batch_size*self.cfg.DATA.VIDEO.NUM_FRAMES, -1, hs, ws)
        F_ta = F.interpolate(F_ta.reshape(batch_size, -1, temper_s, hs, ws), size=(self.cfg.DATA.VIDEO.NUM_FRAMES, hs, ws), mode="trilinear", align_corners=False)
        F_ta = F_ta.reshape(batch_size*self.cfg.DATA.VIDEO.NUM_FRAMES, -1, hs, ws)
        
        # Multi-Resolution Feature Decoder.
        F_vm = self.deconv_vs2m(F_va)
        F_vl = self.deconv_vm2l(F_vm)
        
        m_size = F_vm.shape[-2: ]
        l_size = F_vl.shape[-2: ]
        
        F_tm = F.interpolate(F_ta, size=m_size, mode="bilinear", align_corners=False)
        F_tl = F.interpolate(F_ta, size=l_size, mode="bilinear", align_corners=False)


        text_repr = text_repr.repeat(batch_size*self.cfg.DATA.VIDEO.NUM_FRAMES//text_repr.shape[0], 1, 1)
        # FCN to get segmentation.
        mask_s, bbox_s = self.fcn_s(
            torch.cat([
                self.F_va_max_pool(F_ta)[0].squeeze(2).squeeze(2), 
                self.F_va_max_pool(F_va)[0].squeeze(2).squeeze(2), 
                self.text_max_pool(text_repr.permute(0, 2, 1).contiguous()).squeeze(2)
            ], dim=1), 
            torch.cat([F_va, F_ta, spatial_maps_s.repeat(self.cfg.DATA.VIDEO.NUM_FRAMES//temper_s, 1, 1, 1)], dim=1)
        )
        mask_m, bbox_m = self.fcn_m(
            torch.cat([
                self.F_va_max_pool(F_ta)[0].squeeze(2).squeeze(2), 
                self.F_va_max_pool(F_va)[0].squeeze(2).squeeze(2), 
                self.text_max_pool(text_repr.permute(0, 2, 1).contiguous()).squeeze(2)
            ], dim=1), 
            torch.cat([F_vm, F_tm, spatial_maps_m], dim=1)
        )
        mask_l, bbox_l = self.fcn_l(
            torch.cat([
                self.F_va_max_pool(F_ta)[0].squeeze(2).squeeze(2), 
                self.F_va_max_pool(F_va)[0].squeeze(2).squeeze(2), 
                self.text_max_pool(text_repr.permute(0, 2, 1).contiguous()).squeeze(2)
            ], dim=1), 
            torch.cat([F_vl, F_tl, spatial_maps_l], dim=1)
        )

        l_size = mask_l.shape[-2], mask_l.shape[-1]
        mask_l = self.fcn_fusion(torch.cat([
            utils.resize(mask_s, l_size, False), utils.resize(mask_m, l_size, False), mask_l
        ], dim=1))

        assert bbox_l.shape == bbox_s.shape, "Shape Error"

        # masks = [mask_s.squeeze(1), mask_m.squeeze(1), mask_l.squeeze(1)]
        mask_s, mask_m, mask_l = mask_s.squeeze(1), mask_m.squeeze(1), mask_l.squeeze(1)
        # mask_s = mask_s.reshape(batch_size, self.num_frames, self.resolution[2][0], self.resolution[2][1])
        # mask_m = mask_m.reshape(batch_size, self.num_frames, self.resolution[1][0], self.resolution[1][1])
        # mask_l = mask_l.reshape(batch_size, self.num_frames, self.resolution[0][0], self.resolution[0][1])
        # bbox_s = bbox_s.reshape(batch_size, self.num_frames, 4)
        # bbox_m = bbox_m.reshape(batch_size, self.num_frames, 4)
        # bbox_l = bbox_l.reshape(batch_size, self.num_frames, 4)
        outputs = {
            "mask_s": mask_s, "mask_m": mask_m, "mask_l": mask_l, 
            "bbox_s": bbox_s, "bbox_m": bbox_m, "bbox_l": bbox_l, 
        }
        return outputs


@add_decoder
class MSMFCNDecoderV1(torch.nn.Module):
    """
    This module can deal with multi-task.
    """
    def __init__(self, cfg, *args, **kwargs):
        super(MSMFCNDecoderV1, self).__init__()
        self.cfg = cfg
        self.args = args
        self.kwargs = kwargs
        self.text_repr_dim = self.cfg.DATA.QUERY.DIM
        self.video_repr_channels = self.cfg.DATA.VIDEO.REPR_CHANNELS
        self.spatial_feat_dim = self.cfg.DATA.VIDEO.SPATIAL_FEAT_DIM
        self.resolution = self.cfg.DATA.VIDEO.RESOLUTION
        self.num_frames = self.cfg.DATA.VIDEO.NUM_FRAMES
        self._build_model()

    def _build_model(self):
        self.asymm_cross_attn = AsymmCrossAttnV1(self.text_repr_dim, self.video_repr_channels, self.spatial_feat_dim, self.resolution)
        
        self.deconv_vs2m = nn.Sequential(
            nn.ConvTranspose2d(
                self.video_repr_channels[0], self.video_repr_channels[1], kernel_size=8, stride=4, padding=2, bias=True
            ), 
            nn.Conv2d(
                self.video_repr_channels[1], self.video_repr_channels[1], kernel_size=3, stride=1, padding=1, bias=True
            ),
        )
        self.deconv_vm2l = nn.Sequential(
            nn.ConvTranspose2d(
                self.video_repr_channels[1], self.video_repr_channels[2], kernel_size=8, stride=4, padding=2, bias=True
            ), 
            nn.Conv2d(
                self.video_repr_channels[2], self.video_repr_channels[2], kernel_size=3, stride=1, padding=1, bias=True
            ),
        )

        self.v_gmax = nn.AdaptiveMaxPool2d(1)
        
        self.fcn_s = MFCNModuleV2(self.text_repr_dim, 2*self.text_repr_dim+self.video_repr_channels[0], self.video_repr_channels[0], self.video_repr_channels[1], self.spatial_feat_dim)
        self.fcn_m = MFCNModuleV2(self.text_repr_dim, 2*self.text_repr_dim+self.video_repr_channels[0], self.video_repr_channels[1], self.video_repr_channels[2], self.spatial_feat_dim)
        self.fcn_l = MFCNModuleV2(self.text_repr_dim, 2*self.text_repr_dim+self.video_repr_channels[0], self.video_repr_channels[2], self.video_repr_channels[2]//2, self.spatial_feat_dim)
        
        self.fcn_fusion = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.text_max_pool = nn.AdaptiveMaxPool1d(1)
        self.F_va_max_pool = nn.AdaptiveMaxPool2d(1, 1)
        self.linear_vm = nn.Conv2d(192, self.video_repr_channels[1], kernel_size=1)

    def attend(self, text_repr, video_repr, spatial_maps_s):
        batch_size = video_repr.shape[0]
        v_shape = video_repr.shape[-2], video_repr.shape[-1]

        video_repr = F.normalize(video_repr.mean(-3), p=2, dim=1)

        # (N, 832, 32, 32) + (N, 8, 32, 32) -> (N, 832+8, 32, 32)
        F_vc = torch.cat([video_repr, spatial_maps_s], dim=1)
        
        # video to text attention.
        
        # Align video features to features with same dimension as language features.
        # (N, 832+8, 32, 32) -> (N, 32, 32, 832+8)
        F_vc = F_vc.permute(0, 2, 3, 1).contiguous()
        # (N, 32, 32, 832+8) -> (N*32*32, 832+8) -> (N*32*32, 300)
        F_vc2t = self.linear_vc2t(F_vc.reshape(-1, self.video_repr_channels[0]+self.spatial_map_dim).contiguous())
        # (N*32*32, 300) -> (N, 32*32, 300)
        F_vc2t = F_vc2t.reshape(batch_size, -1, self.text_repr_dim).contiguous()

        # Conducting co-attention
        # (N, 32*32, 300)*[(N, 20, 300) -> (N, 300, 20)] -> (N, 32*32, 20)
        F_ta = torch.matmul(F_vc2t, text_repr.transpose(-1, -2).contiguous())
        F_ta = F_ta * (self.text_repr_dim ** (-0.5))
        F_ta = torch.softmax(F_ta, dim=-1)
        # (N, 32*32, 20)*(N, 20, 300) -> (N, 32*32, 300)
        F_ta = torch.matmul(F_ta, text_repr)
        # (N, 32*32, 300) -> (N, 32, 32, 300)
        F_ta = F_ta.reshape(batch_size, self.resolution[0][0], self.resolution[0][1], self.text_repr_dim).contiguous()
        # (N, 32, 32, 832+8) -> (N, 832+8, 32, 32)
        F_vc = F_vc.permute(0, 3, 1, 2).contiguous()
        # (N, 32, 32, 300) -> (N, 300, 32, 32)
        F_ta = F_ta.permute(0, 3, 1, 2).contiguous()

        # text to video attention

        # Align text features to features with same dimension as visual features.

        F_t2vc = self.linear_t2vc(self.max_pool(text_repr)).repeat(1, self.resolution[0][0]**2, 1).contiguous()

        F_vcq = self.linear_vcq(F_vc.permute(0, 2, 3, 1).contiguous().reshape(batch_size, self.resolution[0][0]**2, self.video_repr_channels[0]+self.spatial_map_dim)).contiguous()
        F_vck = self.linear_vck(F_vc.permute(0, 2, 3, 1).contiguous().reshape(batch_size, self.resolution[0][0]**2, self.video_repr_channels[0]+self.spatial_map_dim)).contiguous()
        F_vcv = self.linear_vcv(F_vc.permute(0, 2, 3, 1).contiguous().reshape(batch_size, self.resolution[0][0]**2, self.video_repr_channels[0]+self.spatial_map_dim)).contiguous()

        F_vcq = F_vcq * F_t2vc
        F_vck = F_vck * F_t2vc

        # Conducting Attention
        F_va = torch.matmul(F_vck, F_vcq.permute(0, 2, 1).contiguous())
        F_va = F_va * ((self.video_repr_channels[0]+self.spatial_map_dim) ** (-0.5))
        F_va = torch.softmax(F_va, dim=-1)
        F_va = torch.matmul(F_va, F_vcv)
        F_va = F_va.reshape(batch_size, self.resolution[0][0], self.resolution[0][1], self.video_repr_channels[0]+self.spatial_map_dim).permute(0, 3, 1, 2).contiguous()
        F_va = F_va[:, :self.video_repr_channels[0], :, :]

        return F_ta, F_va

    def forward(self, text_repr, video_repr, *args, **kwargs):
        spatial_maps = kwargs["spatial_feats"]
        spatial_map_s, spatial_map_m, spatial_map_l = spatial_maps
        batch_size = text_repr.shape[0]
        spatial_maps_s = spatial_map_s.repeat(batch_size, 1, 1, 1)
        spatial_maps_m = spatial_map_m.repeat(batch_size, 1, 1, 1)
        spatial_maps_l = spatial_map_l.repeat(batch_size, 1, 1, 1)
        F_ta, F_va = self.asymm_cross_attn(text_repr, video_repr[-1], spatial_maps_s)

        # Multi-Resolution Feature Decoder.
        F_vm = self.deconv_vs2m(F_va)
        F_vl = self.deconv_vm2l(F_vm)

        m_size = F_vm.shape[-2], F_vm.shape[-1]
        l_size = F_vl.shape[-2], F_vl.shape[-1]

        F_tm = F.interpolate(F_ta, size=m_size, mode="bilinear", align_corners=False)
        F_tl = F.interpolate(F_ta, size=l_size, mode="bilinear", align_corners=False)

        # FCN to get segmentation.
        mask_s, bbox_s = self.fcn_s( 
            torch.cat([
                self.F_va_max_pool(F_ta)[0].squeeze(2).squeeze(2), 
                self.F_va_max_pool(F_va)[0].squeeze(2).squeeze(2), 
                self.text_max_pool(text_repr.permute(0, 2, 1).contiguous()).squeeze(2)
            ], dim=1), 
            torch.cat([F_va + video_repr[-1].mean(-3), spatial_maps_s, F_ta], dim=1)
        )
        mask_m, bbox_m = self.fcn_m(
            torch.cat([
                self.F_va_max_pool(F_ta)[0].squeeze(2).squeeze(2), 
                self.F_va_max_pool(F_va)[0].squeeze(2).squeeze(2), 
                self.text_max_pool(text_repr.permute(0, 2, 1).contiguous()).squeeze(2)
            ], dim=1), 
            torch.cat([F_vm + self.linear_vm(video_repr[-2].mean(-3)), spatial_maps_m, F_tm], dim=1)
        )
        mask_l, bbox_l = self.fcn_l(
            torch.cat([
                self.F_va_max_pool(F_ta)[0].squeeze(2).squeeze(2), 
                self.F_va_max_pool(F_va)[0].squeeze(2).squeeze(2), 
                self.text_max_pool(text_repr.permute(0, 2, 1).contiguous()).squeeze(2)
            ], dim=1), 
            torch.cat([F_vl, spatial_maps_l, F_tl], dim=1)
        )

        l_size = mask_l.shape[-2], mask_l.shape[-1]
        mask_l = self.fcn_fusion(torch.cat([
            utils.resize(mask_s, l_size, False), utils.resize(mask_m, l_size, False), mask_l
        ], dim=1))
        assert bbox_l.shape == bbox_s.shape, "Shape Error"

        mask_s, mask_m, mask_l = mask_s.squeeze(1), mask_m.squeeze(1), mask_l.squeeze(1)
        outputs = {
            "mask_s": mask_s, "mask_m": mask_m, "mask_l": mask_l, 
            "bbox_s": bbox_s, "bbox_m": bbox_m, "bbox_l": bbox_l, 
        }
        return outputs



if __name__ == "__main__":
    print(_DECODER)
    model = _DECODER["UNetDecoder"](None)
    print(_DECODER)