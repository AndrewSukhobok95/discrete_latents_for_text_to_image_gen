import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.common_blocks import ResidualStack, ChangeChannels, DownSampleX2, UpSampleX2

class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_downsamples=2,
                 num_resid_downsample=1,
                 num_resid_bottleneck=4,
                 use_bn=False,
                 use_conv1x1=False):
        super(Encoder, self).__init__()
        self.use_conv1x1 = use_conv1x1
        _out_ch = out_channels // (2**num_downsamples)
        self._change_ch_block = ChangeChannels(in_channels, _out_ch, use_bn=use_bn)
        down_sample_block = []
        for _ in range(num_downsamples):
            _in_ch = _out_ch
            _out_ch = _out_ch * 2
            down_sample_block += [
                DownSampleX2(in_channels=_in_ch, out_channels=_out_ch, use_bn=use_bn),
                ResidualStack(in_channels=_out_ch, out_channels=_out_ch, num_residual_layers=num_resid_downsample, use_bn=use_bn)
            ]
        self._down_sample_block = nn.Sequential(*down_sample_block)
        self._res_block = ResidualStack(in_channels=out_channels, out_channels=out_channels, num_residual_layers=num_resid_bottleneck, use_bn=use_bn)
        self._conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self._change_ch_block(x)
        x = self._down_sample_block(x)
        x = self._res_block(x)
        if self.use_conv1x1:
            x = self._conv1x1(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_upsamples=2,
                 num_resid_upsample=1,
                 num_resid_bottleneck=4,
                 use_bn=False,
                 use_conv1x1=False):
        super(Decoder, self).__init__()
        self.use_conv1x1 = use_conv1x1
        self._conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        _out_ch = in_channels
        self._res_block = ResidualStack(in_channels=in_channels, out_channels=in_channels, num_residual_layers=num_resid_bottleneck, use_bn=use_bn)
        up_sample_block = []
        for _ in range(num_upsamples):
            _in_ch = _out_ch
            _out_ch = _in_ch // 2
            up_sample_block += [
                UpSampleX2(in_channels=_in_ch, out_channels=_out_ch, use_bn=use_bn),
                ResidualStack(in_channels=_out_ch, out_channels=_out_ch, num_residual_layers=num_resid_upsample, use_bn=use_bn)
            ]
        self._up_sample_block = nn.Sequential(*up_sample_block)
        self._change_ch_block = ChangeChannels(_out_ch, out_channels, use_bn=use_bn)

    def forward(self, x):
        if self.use_conv1x1:
            x = self._conv1x1(x)
        x = self._res_block(x)
        x = self._up_sample_block(x)
        x = self._change_ch_block(x)
        return x


