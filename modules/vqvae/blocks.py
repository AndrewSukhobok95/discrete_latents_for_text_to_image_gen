import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, use_bn=False):
        super(Residual, self).__init__()
        self.use_bn = use_bn
        self.conv_1 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, bias=bias)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=bias)
        )

    def forward(self, x):
        x_new = self.conv_1(x)
        if self.use_bn:
            x_new = self.bn(x_new)
        x_new = self.conv_2(x_new)
        return x + x_new


class ResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, bias=False, use_bn=False):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, out_channels, bias, use_bn)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class DownSampleX2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, use_bn=False):
        super(DownSampleX2, self).__init__()
        self.use_bn = use_bn
        self._block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=bias),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self._block(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class UpSampleX2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, use_bn=False):
        super(UpSampleX2, self).__init__()
        self.use_bn = use_bn
        self._block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=bias)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self._block(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class ChangeChannels(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, use_bn=False):
        super(ChangeChannels, self).__init__()
        self.use_bn = use_bn
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, bias=bias),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=bias)
        )

    def forward(self, x):
        x = self.conv_1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.conv_2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_downsamples=2, num_residual_layers=4, use_bn=False, use_conv1x1=False):
        super(Encoder, self).__init__()
        self.use_conv1x1 = use_conv1x1
        _out_ch = out_channels // (2**num_downsamples)
        self._change_ch_block = ChangeChannels(in_channels, _out_ch, use_bn=use_bn)
        down_sample_block = []
        for _ in range(num_downsamples):
            _in_ch = _out_ch
            _out_ch = _out_ch * 2
            down_sample_block += [DownSampleX2(in_channels=_in_ch, out_channels=_out_ch, use_bn=use_bn),
                                  ResidualStack(in_channels=_out_ch, out_channels=_out_ch, num_residual_layers=1, use_bn=use_bn)]
        self._down_sample_block = nn.Sequential(*down_sample_block)
        self._res_block = ResidualStack(in_channels=out_channels, out_channels=out_channels, num_residual_layers=num_residual_layers, use_bn=use_bn)
        self._conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self._change_ch_block(x)
        x = self._down_sample_block(x)
        x = self._res_block(x)
        if self.use_conv1x1:
            x = self._conv1x1(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_upsamples=2, num_residual_layers=4, use_bn=False, use_conv1x1=False):
        super(Decoder, self).__init__()
        self.use_conv1x1 = use_conv1x1
        self._conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        _out_ch = in_channels
        self._res_block = ResidualStack(in_channels=in_channels, out_channels=in_channels, num_residual_layers=num_residual_layers, use_bn=use_bn)
        up_sample_block = []
        for _ in range(num_upsamples):
            _in_ch = _out_ch
            _out_ch = _in_ch // 2
            up_sample_block += [UpSampleX2(in_channels=_in_ch, out_channels=_out_ch, use_bn=use_bn),
                                ResidualStack(in_channels=_out_ch, out_channels=_out_ch, num_residual_layers=1, use_bn=use_bn)]
        self._up_sample_block = nn.Sequential(*up_sample_block)
        self._change_ch_block = ChangeChannels(_out_ch, out_channels, use_bn=use_bn)

    def forward(self, x):
        if self.use_conv1x1:
            x = self._conv1x1(x)
        x = self._res_block(x)
        x = self._up_sample_block(x)
        x = self._change_ch_block(x)
        return x


