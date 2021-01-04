
"""
Author  Yiqun Chen
Docs    Decoder classes.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

_DECODER = {}

def add_decoder(_class):
    _DECODER[_class.__name__] = _class
    return _class

@add_decoder
class UNetDecoder(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(UNetDecoder, self).__init__()
        self.cfg = cfg
        self._build_model()

    def _build_model(self):
        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet', 
            in_channels=3, out_channels=1, init_features=32, pretrained=True
        )
        self.model = nn.ModuleDict({
            "decoder4": model.decoder4, 
            "decoder3": model.decoder3,
            "decoder2": model.decoder2,
            "decoder1": model.decoder1,
            "upconv4": model.upconv4, 
            "upconv3": model.upconv3,
            "upconv2": model.upconv2,
            "upconv1": model.upconv1,   
            "conv": model.conv, 
        })
        
    def forward(self, data, *args, **kwargs):
        enc1, enc2, enc3, enc4, bottleneck = data
        dec4 = self.model["upconv4"](bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.model["decoder4"](dec4)
        dec3 = self.model["upconv3"](dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.model["decoder3"](dec3)
        dec2 = self.model["upconv2"](dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.model["decoder2"](dec2)
        dec1 = self.model["upconv1"](dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.model["decoder1"](dec1)
        raise NotImplementedError("Method UNetDecoder.forward is not implemented.")



if __name__ == "__main__":
    print(_DECODER)
    model = _DECODER["UNetDecoder"](None)
    print(_DECODER)