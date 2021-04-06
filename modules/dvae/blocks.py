import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.common_blocks import ResidualStack, ChangeChannels


class DownSampleX2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, use_bn=True):
        super(DownSampleX2, self).__init__()
        self.use_bn = use_bn
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.conv_1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.conv_2(F.relu(x))
        return F.relu(x)


class UpSampleX2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, use_bn=True):
        super(UpSampleX2, self).__init__()
        self.use_bn = use_bn
        self.tconv_1 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.tconv_2 = nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.tconv_1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.tconv_2(F.relu(x))
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_x2downsamples=2,
                 num_resids_downsample=2,
                 num_resids_bottleneck=2,
                 bias=True,
                 use_bn=True):
        super(Encoder, self).__init__()
        _out_ch = out_channels // (2**num_x2downsamples)
        self._change_ch_block = ChangeChannels(in_channels, _out_ch,
                                               bias=bias, use_bn=use_bn)

        down_sample_block = []
        for _ in range(num_x2downsamples):
            _in_ch = _out_ch
            _out_ch = _out_ch * 2
            down_sample_block += [
                DownSampleX2(in_channels=_in_ch, out_channels=_out_ch,
                             bias=bias, use_bn=use_bn),
                ResidualStack(in_channels=_out_ch, out_channels=_out_ch,
                              num_residual_layers=num_resids_downsample,
                              bias=bias, use_bn=use_bn)
            ]
        self._down_sample_block = nn.Sequential(*down_sample_block)
        self._res_block = ResidualStack(in_channels=out_channels, out_channels=out_channels,
                                        num_residual_layers=num_resids_bottleneck,
                                        bias=bias, use_bn=use_bn)

    def forward(self, x):
        x = self._change_ch_block(x)
        x = self._down_sample_block(x)
        x = self._res_block(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_x2upsamples=2,
                 num_resids_upsample=2,
                 num_resids_bottleneck=2,
                 bias=True,
                 use_bn=True):
        super(Decoder, self).__init__()
        _out_ch = in_channels
        up_sample_block = []
        for _ in range(num_x2upsamples):
            _in_ch = _out_ch
            _out_ch = _in_ch // 2
            up_sample_block += [
                UpSampleX2(in_channels=_in_ch, out_channels=_out_ch,
                           bias=bias, use_bn=use_bn),
                ResidualStack(in_channels=_out_ch, out_channels=_out_ch,
                              num_residual_layers=num_resids_upsample,
                              bias=bias, use_bn=use_bn)
            ]
        self._res_block = ResidualStack(in_channels=in_channels, out_channels=in_channels,
                                        num_residual_layers=num_resids_bottleneck,
                                        bias=bias, use_bn=use_bn)
        self._up_sample_block = nn.Sequential(*up_sample_block)
        self._change_ch_block = ChangeChannels(_out_ch, out_channels, use_bn=use_bn)

    def forward(self, x):
        x = self._res_block(x)
        x = self._up_sample_block(x)
        x = self._change_ch_block(x)
        return torch.sigmoid(x)


