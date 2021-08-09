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
    def __init__(self, in_channels, out_channels, num_residual_layers, bias=False, use_bn=False, final_relu=True):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self.final_relu = final_relu
        self._layers = nn.ModuleList([Residual(in_channels, out_channels, bias, use_bn)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        if self.final_relu:
            return F.relu(x)
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


class TrEncoderBlock(nn.Module):
    def __init__(self, n_features, n_attn_heads, n_hidden=64, dropout_prob=0.1, norm_first=True):
        super(TrEncoderBlock, self).__init__()
        self.norm_first = norm_first

        self.attn = nn.MultiheadAttention(n_features, n_attn_heads)
        self.ln1 = nn.LayerNorm(n_features)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Dropout(dropout_prob),
            nn.GELU(),
            nn.Linear(n_hidden, n_features)
        )
        self.ln2 = nn.LayerNorm(n_features)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x, pad_mask=None, attn_mask=None):
        if self.norm_first:
            xn = self.ln1(x)
            dx, _ = self.attn(query=xn, key=xn, value=xn,
                              key_padding_mask=pad_mask,
                              attn_mask=attn_mask)
            x = x + self.dropout1(dx)
            xn = self.ln2(x)
            dx = self.mlp(xn)
            x = x + self.dropout2(dx)
        else:
            dx, _ = self.attn(query=x, key=x, value=x,
                              key_padding_mask=pad_mask,
                              attn_mask=attn_mask)
            x = self.ln1(x + self.dropout1(dx))
            dx = self.mlp(x)
            x = self.ln2(x + self.dropout2(dx))
        return x


class TrDecoderBlock(nn.Module):
    def __init__(self, n_features, n_attn_heads, n_hidden=64, dropout_prob=0.1, norm_first=True):
        super(TrDecoderBlock, self).__init__()
        self.norm_first = norm_first

        self.self_attn = nn.MultiheadAttention(n_features, n_attn_heads)
        self.ln1 = nn.LayerNorm(n_features)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.cross_attn = nn.MultiheadAttention(n_features, n_attn_heads)
        self.ln2 = nn.LayerNorm(n_features)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features)
        )
        self.ln3 = nn.LayerNorm(n_features)
        self.dropout3 = nn.Dropout(dropout_prob)

    def forward(self, x, y, pad_mask=None, attn_mask=None):
        if self.norm_first:
            xn = self.ln1(x)
            dx, self_attn_map = self.self_attn(query=xn, key=xn, value=xn, attn_mask=attn_mask)
            x = x + self.dropout1(dx)

            xn = self.ln2(x)
            dx, cross_attn_map = self.cross_attn(query=xn, key=y, value=y, key_padding_mask=pad_mask)
            x = x + self.dropout2(dx)

            xn = self.ln3(x)
            dx = self.mlp(xn)
            x = x + self.dropout3(dx)
        else:
            dx, self_attn_map = self.self_attn(query=x, key=x, value=x, attn_mask=attn_mask)
            x = self.ln1(x + self.dropout1(dx))

            dx, cross_attn_map = self.cross_attn(query=x, key=y, value=y, key_padding_mask=pad_mask)
            x = self.ln2(x + self.dropout2(dx))

            dx = self.mlp(x)
            x = self.ln3(x + self.dropout3(dx))
        return x, self_attn_map, cross_attn_map



