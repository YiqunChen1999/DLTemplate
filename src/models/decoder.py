
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

_DECODER = {}

def add_decoder(decoder):
    _DECODER[decoder.__name__] = decoder
    return decoder

@add_decoder
class MFMFCNDecoderV2(torch.nn.Module):
    """
    This module can deal with multi-task.
    """
    def __init__(self, cfg, *args, **kwargs):
        super(MFMFCNDecoderV2, self).__init__()
        self.cfg = cfg
        self.args = args
        self.kwargs = kwargs
        self.text_repr_dim = self.cfg.DATA.SENTENCES.DIM
        self.video_repr_channels = self.cfg.DATA.VIDEO.REPR_CHANNELS
        self.spatial_map_dim = self.cfg.DATA.VIDEO.SPATIAL_MAP_DIM
        self.class_num = self.cfg.DATA.VIDEO.CLASS_NUM
        self.resolution = self.cfg.DATA.VIDEO.RESOLUTION
        self._build_model()

    def _build_model(self):
        # Linear transformation from video to text.
        self.linear_vc2t = nn.Linear(self.video_repr_channels[0]+self.spatial_map_dim, self.text_repr_dim, bias=True)
        self.max_pool = nn.AdaptiveMaxPool2d((1, self.text_repr_dim))
        self.linear_t2vc = nn.Linear(self.text_repr_dim, self.video_repr_channels[0]+self.spatial_map_dim, bias=True)

        self.linear_vck = nn.Linear(self.video_repr_channels[0]+self.spatial_map_dim, self.video_repr_channels[0]+self.spatial_map_dim, bias=True)
        self.linear_vcq = nn.Linear(self.video_repr_channels[0]+self.spatial_map_dim, self.video_repr_channels[0]+self.spatial_map_dim, bias=True)
        self.linear_vcv = nn.Linear(self.video_repr_channels[0]+self.spatial_map_dim, self.video_repr_channels[0]+self.spatial_map_dim, bias=True)
        
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
        
        self.fcn_s = nn.ModuleDict({
            "conv_1": nn.Conv2d(
                self.text_repr_dim+self.video_repr_channels[0]+self.spatial_map_dim, self.video_repr_channels[1], kernel_size=3, stride=1, padding=1, bias=True
            ), 
            "cbn_1": ConditionalBatchNorm2d(
                2*self.text_repr_dim+self.video_repr_channels[0], self.video_repr_channels[0], self.video_repr_channels[1]
            ), 
            "relu_1": nn.ReLU(), 
            "conv_2": nn.Conv2d(
                self.video_repr_channels[1], self.video_repr_channels[1], kernel_size=3, stride=1, padding=1, bias=True
            ), 
            "cbn_2": ConditionalBatchNorm2d(
                2*self.text_repr_dim+self.video_repr_channels[0], self.video_repr_channels[0], self.video_repr_channels[1]
            ), 
            "relu_2": nn.ReLU(), 
            "conv_3": nn.Conv2d(
                self.video_repr_channels[1], 1, kernel_size=1, stride=1, bias=True
            ), 
            "box_branch": nn.Sequential(
                nn.Linear(self.video_repr_channels[1], 4), 
            ), 
        })
        self.fcn_m = nn.ModuleDict({
            "conv_1": nn.Conv2d(
                self.text_repr_dim+self.video_repr_channels[1]+self.spatial_map_dim, self.video_repr_channels[2], kernel_size=3, stride=1, padding=1, bias=True
            ), 
            "cbn_1": ConditionalBatchNorm2d(
                2*self.text_repr_dim+self.video_repr_channels[0], self.video_repr_channels[0], self.video_repr_channels[2]
            ), 
            "relu_1": nn.ReLU(), 
            "conv_2": nn.Conv2d(
                self.video_repr_channels[2], self.video_repr_channels[2], kernel_size=3, stride=1, padding=1, bias=True
            ), 
            "cbn_2": ConditionalBatchNorm2d(
                2*self.text_repr_dim+self.video_repr_channels[0], self.video_repr_channels[0], self.video_repr_channels[2]
            ), 
            "relu_2": nn.ReLU(), 
            "conv_3": nn.Conv2d(
                self.video_repr_channels[2], 1, kernel_size=1, stride=1, bias=True
            ), 
            "box_branch": nn.Sequential(
                nn.Linear(self.video_repr_channels[2], 4), 
            ), 
        })
        self.fcn_l = nn.ModuleDict({
            "conv_1": nn.Conv2d(
                self.text_repr_dim+self.video_repr_channels[2]+self.spatial_map_dim, self.video_repr_channels[2]//2, kernel_size=3, stride=1, padding=1, bias=True
            ), 
            "cbn_1": ConditionalBatchNorm2d(
                2*self.text_repr_dim+self.video_repr_channels[0], self.video_repr_channels[0], self.video_repr_channels[2]//2
            ), 
            "relu_1": nn.ReLU(), 
            "conv_2": nn.Conv2d(
                self.video_repr_channels[2]//2, self.video_repr_channels[2]//2, kernel_size=3, stride=1, padding=1, bias=True
            ), 
            "cbn_2": ConditionalBatchNorm2d(
                2*self.text_repr_dim+self.video_repr_channels[0], self.video_repr_channels[0], self.video_repr_channels[2]//2
            ), 
            "relu_2": nn.ReLU(), 
            "conv_3": nn.Conv2d(
                self.video_repr_channels[2]//2, 1, kernel_size=1, stride=1, bias=True
            ), 
            "box_branch": nn.Sequential(
                nn.Linear(self.video_repr_channels[2]//2, 4), 
            ), 
        })
        
        self.fcn_fusion = nn.Conv2d(
            3, 1, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.text_max_pool = nn.AdaptiveMaxPool1d(1)
        self.F_va_max_pool = nn.AdaptiveMaxPool2d(1, 1)
        self.linear_vm = nn.Conv2d(192, self.video_repr_channels[1], kernel_size=1)
        self.linear = nn.Linear(self.video_repr_channels[0], self.text_repr_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.box_fusion = nn.Conv1d(3, 1, 1)

    def fcn(self, fcn_func, text_repr, video_repr):
        # NOTE 
        video_repr = fcn_func["conv_1"](video_repr)
        video_repr, _ = fcn_func["cbn_1"](text_repr, video_repr)
        video_repr = fcn_func["relu_1"](video_repr)
        video_repr = fcn_func["conv_2"](video_repr)
        video_repr, _ = fcn_func["cbn_2"](text_repr, video_repr)
        video_vec = self.v_gmax(video_repr).squeeze(2).squeeze(2)
        bbox = fcn_func["box_branch"](video_vec)
        video_repr = fcn_func["relu_2"](video_repr)
        video_repr = fcn_func["conv_3"](video_repr)
        return video_repr, bbox

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
        spatial_maps = kwargs["spatial_maps"]
        spatial_map_s, spatial_map_m, spatial_map_l = spatial_maps
        batch_size = text_repr.shape[0]
        spatial_maps_s = spatial_map_s.repeat(batch_size, 1, 1, 1)
        spatial_maps_m = spatial_map_m.repeat(batch_size, 1, 1, 1)
        spatial_maps_l = spatial_map_l.repeat(batch_size, 1, 1, 1)
        F_ta, F_va = self.attend(text_repr, video_repr[-1], spatial_maps_s)

        # Multi-Resolution Feature Decoder.
        F_vm = self.deconv_vs2m(F_va)
        F_vl = self.deconv_vm2l(F_vm)

        m_size = F_vm.shape[-2], F_vm.shape[-1]
        l_size = F_vl.shape[-2], F_vl.shape[-1]

        F_tm = F.interpolate(F_ta, size=m_size, mode="bilinear", align_corners=False)
        F_tl = F.interpolate(F_ta, size=l_size, mode="bilinear", align_corners=False)

        # FCN to get segmentation.
        pred_s, bbox_s = self.fcn(
            self.fcn_s, 
            torch.cat([
                self.F_va_max_pool(F_ta)[0].squeeze(2).squeeze(2), 
                self.F_va_max_pool(F_va)[0].squeeze(2).squeeze(2), 
                self.text_max_pool(text_repr.permute(0, 2, 1).contiguous()).squeeze(2)
            ], dim=1), 
            torch.cat([F_va + video_repr[-1].mean(-3), spatial_maps_s, F_ta], dim=1)
        )
        pred_m, bbox_m = self.fcn(
            self.fcn_m, 
            torch.cat([
                self.F_va_max_pool(F_ta)[0].squeeze(2).squeeze(2), 
                self.F_va_max_pool(F_va)[0].squeeze(2).squeeze(2), 
                self.text_max_pool(text_repr.permute(0, 2, 1).contiguous()).squeeze(2)
            ], dim=1), 
            torch.cat([F_vm + self.linear_vm(video_repr[-2].mean(-3)), spatial_maps_m, F_tm], dim=1)
        )
        pred_l, bbox_l = self.fcn(
            self.fcn_l, 
            torch.cat([
                self.F_va_max_pool(F_ta)[0].squeeze(2).squeeze(2), 
                self.F_va_max_pool(F_va)[0].squeeze(2).squeeze(2), 
                self.text_max_pool(text_repr.permute(0, 2, 1).contiguous()).squeeze(2)
            ], dim=1), 
            torch.cat([F_vl, spatial_maps_l, F_tl], dim=1)
        )

        l_size = pred_l.shape[-2], pred_l.shape[-1]
        pred_l = self.fcn_fusion(torch.cat([
            utils.resize(pred_s, l_size, False), utils.resize(pred_m, l_size, False), pred_l
        ], dim=1))
        # bbox_l = self.box_fusion(torch.cat([
        #     bbox_s.unsqueeze(1), bbox_m.unsqueeze(1), bbox_l.unsqueeze(1)
        # ], dim=1)).squeeze(1)
        assert bbox_l.shape == bbox_s.shape, "Shape Error"

        # preds = [pred_s.squeeze(1), pred_m.squeeze(1), pred_l.squeeze(1)]
        pred_s, pred_m, pred_l = pred_s.squeeze(1), pred_m.squeeze(1), pred_l.squeeze(1)
        return pred_s, pred_m, pred_l, bbox_s, bbox_m, bbox_l



if __name__ == "__main__":
    print(_DECODER)
    model = _DECODER["UNetDecoder"](None)
    print(_DECODER)