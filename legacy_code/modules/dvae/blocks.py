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
                 hidden_channels=None,
                 bias=True,
                 use_bn=True):
        super(Encoder, self).__init__()

        self.model_architecture = "Encoder Model:\n"

        if hidden_channels is None:
            _out_ch = out_channels // (2 ** num_x2downsamples)
        else:
            _out_ch = hidden_channels // (2 ** num_x2downsamples)

        self._change_ch_block_start = ChangeChannels(in_channels, _out_ch,
                                                     bias=bias, use_bn=use_bn)
        self.model_architecture += "  ChangeChannels(in={}, out={})\n".format(in_channels, _out_ch)

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
            self.model_architecture += "  DownSampleX2(in={}, out={})\n".format(_in_ch, _out_ch)
            self.model_architecture += "  ResidualStack(in={}, out={}, n_resid={})\n".format(_out_ch, _out_ch,
                                                                                             num_resids_downsample)

        self._down_sample_block = nn.Sequential(*down_sample_block)
        self._res_block = ResidualStack(in_channels=_out_ch, out_channels=_out_ch,
                                        num_residual_layers=num_resids_bottleneck,
                                        bias=bias, use_bn=use_bn)
        self._change_ch_block_end = ChangeChannels(_out_ch, out_channels,
                                                   bias=bias, use_bn=use_bn)

        self.model_architecture += "  ResidualStack(in={}, out={}, n_resid={})\n".format(_out_ch, _out_ch,
                                                                                         num_resids_bottleneck)
        self.model_architecture += "  DownSampleX2(in={}, out={})\n".format(_out_ch, out_channels)

    def forward(self, x):
        x = self._change_ch_block_start(x)
        x = self._down_sample_block(x)
        x = self._res_block(x)
        x = self._change_ch_block_end(x)
        return x

    def show_model_architecture(self):
        print(self.model_architecture)
        # with open('filename.txt', 'w') as f:
        #     print('This message will be written to a file.', file=f)


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_x2upsamples=2,
                 num_resids_upsample=2,
                 num_resids_bottleneck=2,
                 hidden_channels=None,
                 bias=True,
                 use_bn=True):
        super(Decoder, self).__init__()

        self.model_architecture = "Decoder Model:\n"

        if hidden_channels is None:
            _out_ch = in_channels
        else:
            _out_ch = hidden_channels

        self._change_ch_block_start = ChangeChannels(in_channels, _out_ch,
                                                     bias=bias, use_bn=use_bn)
        self._res_block = ResidualStack(in_channels=_out_ch, out_channels=_out_ch,
                                        num_residual_layers=num_resids_bottleneck,
                                        bias=bias, use_bn=use_bn)

        self.model_architecture += "  ChangeChannels(in={}, out={})\n".format(in_channels, _out_ch)
        self.model_architecture += "  ResidualStack(in={}, out={}, n_resid={})\n".format(_out_ch, _out_ch,
                                                                                         num_resids_bottleneck)

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

            self.model_architecture += "  DownSampleX2(in={}, out={})\n".format(_in_ch, _out_ch)
            self.model_architecture += "  ResidualStack(in={}, out={}, n_resid={})\n".format(_out_ch, _out_ch,
                                                                                             num_resids_upsample)

        self._up_sample_block = nn.Sequential(*up_sample_block)
        self._change_ch_block_end = ChangeChannels(_out_ch, out_channels,
                                                   bias=bias, use_bn=use_bn)

        self.model_architecture += "  ChangeChannels(in={}, out={})\n".format(_out_ch, out_channels)

    def forward(self, x):
        x = self._change_ch_block_start(x)
        x = self._res_block(x)
        x = self._up_sample_block(x)
        x = self._change_ch_block_end(x)
        return torch.sigmoid(x)

    def show_model_architecture(self):
        print(self.model_architecture)
        # with open('filename.txt', 'w') as f:
        #     print('This message will be written to a file.', file=f)


