import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, use_bn=False):
        super(Residual, self).__init__()
        self.use_bn = use_bn
        self.conv_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, bias=bias)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Sequential(
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=bias)
        )

    def forward(self, x):
        x = self.conv_1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.conv_2(x)
        return x


