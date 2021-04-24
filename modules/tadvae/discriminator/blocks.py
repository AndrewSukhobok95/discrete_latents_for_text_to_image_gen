import torch
import torch.nn as nn
import torch.nn.functional as F


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



