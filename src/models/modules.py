
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


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, eps=1e-5, momentum=0.9):
        super(ConditionalBatchNorm2d, self).__init__()
        self.in_size, self.hidden_size, self.out_size, self.eps, self.momentum = \
            in_size, hidden_size, out_size, eps, momentum
        self._build_model()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def _build_model(self):
        self.weight = nn.Parameter(torch.ones(self.out_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.out_size), requires_grad=True)
        self.running_mean = nn.Parameter(torch.zeros(self.out_size), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(self.out_size), requires_grad=False)
        self.num_batches_tracked = torch.zeros(1, dtype=torch.long)
        self.fc_weight = nn.Sequential(
                nn.Linear(self.in_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.out_size),
            )

        self.fc_bias = nn.Sequential(
                nn.Linear(self.in_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.out_size),
            )

    def forward(self, text_repr, video_repr):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        bn_training = True if self.training else False

        delta_weight = self.fc_weight(text_repr)
        delta_bias = self.fc_bias(text_repr)

        video_repr, running_mean, running_var = self.batch_norm(
            video_repr, self.running_mean, self.running_var, self.weight+delta_weight, \
                self.bias+delta_bias, self.training, exponential_average_factor, self.eps
        )
        self.running_mean.data, self.running_var.data = running_mean.data, running_var.data
        return video_repr, text_repr

    def batch_norm(
            self, input, running_mean, running_var, weight, bias, is_training, \
                exponential_average_factor, eps
        ):
        # Extract the dimensions
        N, C, H, W = input.size()

        # Mini-batch mean and variance
        # input_channel_major = input.permute(1, 0, 2, 3).contiguous().view(input.size(1), -1)
        # mean = input_channel_major.mean(dim=1)
        # variance = input_channel_major.var(dim=1)
        mean = input.mean(dim=[0, 2, 3])
        variance = input.var(dim=[0, 2, 3])

        # Normalize
        if is_training:
            
            #Compute running mean and variance
            running_mean = running_mean*(1-exponential_average_factor) + mean*exponential_average_factor
            running_var = running_var*(1-exponential_average_factor) + variance*exponential_average_factor
        
            # Training mode, normalize the data using its mean and variance
            X_hat = (input - mean.view(1,C,1,1)) * 1.0 / torch.sqrt(variance.view(1,C,1,1) + eps)
        else:
            # Test mode, normalize the data using the running mean and variance
            X_hat = (input - running_mean.view(1,C,1,1)) * 1.0 / torch.sqrt(running_var.view(1,C,1,1) + eps)
                 
        # Scale and shift
        out = weight.contiguous().view(N,C,1,1) * X_hat + bias.contiguous().view(N,C,1,1)
        
        return out, running_mean, running_var
        

class ConditionalBatchNorm3d(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, eps=1e-5, momentum=0.9):
        super(ConditionalBatchNorm3d, self).__init__()
        self.in_size, self.hidden_size, self.out_size, self.eps, self.momentum = \
            in_size, hidden_size, out_size, eps, momentum
        self._build_model()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def _build_model(self):
        self.weight = nn.Parameter(torch.ones(self.out_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.out_size), requires_grad=True)
        self.running_mean = nn.Parameter(torch.zeros(self.out_size), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(self.out_size), requires_grad=False)
        self.num_batches_tracked = torch.zeros(1, dtype=torch.long)
        self.fc_weight = nn.Sequential(
            nn.Linear(self.in_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.out_size),
            # nn.Tanh(), 
        )
        self.fc_bias = nn.Sequential(
            nn.Linear(self.in_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.out_size),
            # nn.Tanh(), 
        )

    def forward(self, text_repr, video_repr):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        bn_training = True if self.training else False

        delta_weight = self.fc_weight(text_repr)
        delta_bias = self.fc_bias(text_repr)

        video_repr, running_mean, running_var = self.batch_norm(
            video_repr, self.running_mean, self.running_var, self.weight+delta_weight, \
                self.bias+delta_bias, self.training, exponential_average_factor, self.eps
        )
        self.running_mean.data, self.running_var.data = running_mean.data, running_var.data
        return video_repr, text_repr

    def batch_norm(
            self, input, running_mean, running_var, weight, bias, is_training, \
                exponential_average_factor, eps
        ):
        # Extract the dimensions
        N, C, T, H, W = input.size()

        # Mini-batch mean and variance
        mean = input.mean(dim=[0, 2, 3, 4])
        variance = input.var(dim=[0, 2, 3, 4])

        # Normalize
        if is_training:
            
            #Compute running mean and variance
            running_mean = running_mean*(1-exponential_average_factor) + mean*exponential_average_factor
            running_var = running_var*(1-exponential_average_factor) + variance*exponential_average_factor
        
            # Training mode, normalize the data using its mean and variance
            # X_hat = (input - mean.view(1,C,1,1,1).expand((N, C, T, H, W))) * 1.0 / torch.sqrt(variance.view(1,C,1,1,1).expand((N, C, T, H, W)) + eps)
            X_hat = (input - mean.view(1,C,1,1,1)) * 1.0 / torch.sqrt(variance.view(1,C,1,1,1) + eps)
        else:
            # Test mode, normalize the data using the running mean and variance
            # X_hat = (input - running_mean.view(1,C,1,1,1).expand((N, C, T, H, W))) * 1.0 / torch.sqrt(running_var.view(1,C,1,1,1).expand((N, C, T, H, W)) + eps)
            X_hat = (input - running_mean.view(1,C,1,1,1)) * 1.0 / torch.sqrt(running_var.view(1,C,1,1,1) + eps)
                 
        # Scale and shift
        # out = weight.contiguous().view(N,C,1,1,1).expand((N, C, T, H, W)) * X_hat + bias.contiguous().view(N,C,1,1,1).expand((N, C, T, H, W))
        out = weight.contiguous().view(N,C,1,1,1) * X_hat + bias.contiguous().view(N,C,1,1,1)
        
        return out, running_mean, running_var


class TextVector(nn.Module):
    """
    Transform a text matrix into a vector.
    """
    def __init__(self, t_in_channels, t_out_channels, *args, **kwargs):
        super(TextVector, self).__init__()
        self.t_in_channels = t_in_channels
        self.t_out_channels = t_out_channels
        self.reservation = 4
        self._build_model()

    def _build_model(self):
        self.tanh = nn.Tanh()
        self.max_pool = nn.AdaptiveMaxPool1d(self.reservation)
        self.linear_1 = nn.Linear(self.t_in_channels, self.t_in_channels)
        # self.conv = nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1)
        self.linear_2 = nn.Linear(self.reservation*self.t_in_channels, self.t_out_channels)

    def forward(self, text_repr):
        batch_size = text_repr.shape[0]
        text_repr = self.tanh(self.linear_1(text_repr))
        text_repr = text_repr.permute(0, 2, 1).contiguous()
        text_repr = self.max_pool(text_repr)
        text_repr = text_repr.permute(0, 2, 1).contiguous()
        text_repr = text_repr.reshape(batch_size, self.reservation*self.t_in_channels)
        text_repr = self.tanh(self.linear_2(text_repr))
        text_repr = text_repr.unsqueeze(2)
        return text_repr


class EarlyFusion(nn.Module):
    def __init__(self, t_in_channels, t_out_channels, *args, **kwargs):
        super(EarlyFusion, self).__init__()
        self.t_in_channels = t_in_channels
        self.t_out_channels = t_out_channels
        self._build_model()

    def _build_model(self):
        self.t_linear = nn.Linear(self.t_in_channels, self.t_out_channels)
        self.t_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, text_repr, video_repr):
        text_repr = self.t_linear(text_repr)
        text_repr = text_repr.permute(0, 2, 1).contiguous()
        text_repr = self.t_max_pool(text_repr)
        assert len(text_repr.shape) == 3, "Dimension Error"
        text_repr = text_repr.unsqueeze(3).unsqueeze(4)
        video_repr = video_repr * text_repr + video_repr
        return video_repr


class EarlyFusionWithCBN(nn.Module):
    def __init__(self, t_in_channels, t_out_channels, *args, **kwargs):
        super(EarlyFusionWithCBN, self).__init__()
        self.t_in_channels = t_in_channels
        self.t_out_channels = t_out_channels
        self._build_model()

    def _build_model(self):
        self.t_linear = nn.Linear(self.t_in_channels, self.t_out_channels)
        self.text_vector = TextVector(self.t_in_channels, self.t_out_channels)
        self.cbn = ConditionalBatchNorm3d(self.t_out_channels+self.t_out_channels, self.t_out_channels, self.t_out_channels)
        # self.cbn = ConditionalBatchNorm2d(self.t_out_channels+self.t_out_channels, self.t_out_channels, self.t_out_channels)
        self.t_max_pool = nn.AdaptiveMaxPool1d(1)
        self.v_max_pool = nn.AdaptiveAvgPool2d(1)
        self.linear_vec = nn.Linear(self.t_out_channels+self.t_out_channels, self.t_out_channels+self.t_out_channels)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_repr, video_repr):
        text_repr = self.t_linear(text_repr)
        text_repr = text_repr.permute(0, 2, 1).contiguous()
        text_repr = self.t_max_pool(text_repr)
        assert len(text_repr.shape) == 3, "Dimension Error"

        _video_repr = video_repr * text_repr.unsqueeze(3).unsqueeze(4) + video_repr
        # video_repr = self.relu(video_repr)
        # _video_repr = F.normalize(video_repr.mean(-3), p=2, dim=1)
        text_repr = torch.cat([
            text_repr.squeeze(2), 
            self.v_max_pool(F.normalize(video_repr.mean(-3), p=2, dim=1)).squeeze(3).squeeze(2)
        ], dim=1)
        # text_repr = self.relu(self.linear_vec(text_repr))
        # text_repr = self.linear_vec(text_repr)
        video_repr = self.cbn(text_repr, _video_repr)[0] + _video_repr
        
        return video_repr


class _Inflated3DConvNet(nn.Module):
    """
    Basic Inflated 3D ConvNet.
    """
    def __init__(self, num_classes=400, modality="RGB", dropout_prob=0, *args, **kwargs):
        super(_Inflated3DConvNet, self).__init__()
        self.num_classes = num_classes
        self.modality = modality
        self.dropout_prob = dropout_prob
        self.args = args
        self.kwargs = kwargs
        self._build_model()

    def _build_model(self):
        if self.modality == 'RGB':
            in_channels = 3
        elif self.modality == 'Flow':
            in_channels = 2
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(self.modality))
        
        # 1st conv-pool
        self.conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64, in_channels=in_channels, kernel_size=(7, 7, 7), stride=(2, 2, 2), \
                padding='SAME'
        )
        
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME'
        )
        # conv conv
        self.conv3d_2b_1x1 = Unit3Dpy(
            out_channels=64, in_channels=64, kernel_size=(1, 1, 1), padding='SAME'
        )
        
        self.conv3d_2c_3x3 = Unit3Dpy(
            out_channels=192, in_channels=64, kernel_size=(3, 3, 3), padding='SAME'
        )
        
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME'
        )

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME'
        )

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME'
        )

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = torch.nn.Dropout(self.dropout_prob)
        self.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024, out_channels=self.num_classes, kernel_size=(1, 1, 1), \
                activation=None, use_bias=True, use_bn=False
        )
        self.softmax = torch.nn.Softmax(1)

    def forward(self, frames):
        # TODO 
        raise NotImplementedError("Method _Inflated3DConvNet.forward is not implemented yet.")


# ################################
# I3D Modules                    #
# ################################
"""
I3D from Asymmetric Attention code. 
functions `get_padding_shape`, `simplify_padding` and class `Unit3Dpy`, 'MaxPool3dTFPadding', 
'Mixed' are used in Inflated 3D ConvNet
"""
def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class Unit3Dpy(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == 'relu':
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(
            in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(
            in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(
            in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(
            in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out




