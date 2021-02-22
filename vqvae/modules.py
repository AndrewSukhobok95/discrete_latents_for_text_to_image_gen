import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, out_channels)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class DownSampleX2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleX2, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self._block(x)


class UpSampleX2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleX2, self).__init__()
        self._block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        )

    def forward(self, x):
        return self._block(x)


class ChangeChannels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChangeChannels, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return self._block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_downsamples=2):
        super(Encoder, self).__init__()
        _out_ch = out_channels // (2**num_downsamples)
        self._change_ch_block = ChangeChannels(in_channels, _out_ch)
        down_sample_block = []
        for _ in range(num_downsamples):
            _in_ch = _out_ch
            _out_ch = _out_ch * 2
            down_sample_block += [DownSampleX2(in_channels=_in_ch, out_channels=_out_ch),
                                  ResidualStack(in_channels=_out_ch, out_channels=_out_ch, num_residual_layers=1)]
        self._down_sample_block = nn.Sequential(*down_sample_block)
        self._res_block = ResidualStack(in_channels=out_channels, out_channels=out_channels, num_residual_layers=2)

    def forward(self, x):
        x = self._change_ch_block(x)
        x = self._down_sample_block(x)
        return self._res_block(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_upsamples=2):
        super(Decoder, self).__init__()
        _out_ch = in_channels
        self._res_block = ResidualStack(in_channels=in_channels, out_channels=in_channels, num_residual_layers=2)
        up_sample_block = []
        for _ in range(num_upsamples):
            _in_ch = _out_ch
            _out_ch = _in_ch // 2
            up_sample_block += [UpSampleX2(in_channels=_in_ch, out_channels=_out_ch),
                                ResidualStack(in_channels=_out_ch, out_channels=_out_ch, num_residual_layers=1)]
        self._up_sample_block = nn.Sequential(*up_sample_block)
        self._change_ch_block = ChangeChannels(_out_ch, out_channels)

    def forward(self, x):
        x = self._res_block(x)
        x = self._up_sample_block(x)
        return self._change_ch_block(x)


