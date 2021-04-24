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


class TrEncoderBlock(nn.Module):
    def __init__(self, n_features, n_attn_heads, n_hidden=64, dropout_prob=0.1):
        """
        :param n_features:    Number of input and output features.
        :param n_attn_heads:  Number of attention heads in the Multi-Head Attention.
        :param n_hidden:      Number of hidden units in the Feedforward (MLP) block.
        :param dropout_prob:  Dropout rate after the first layer of the MLP and in two places on the main path
                              (before combining the main path with a skip connection).
        """
        super(TrEncoderBlock, self).__init__()
        self.n_features = n_features

        self.attn = nn.MultiheadAttention(n_features, n_attn_heads)
        self.ln1 = nn.LayerNorm(n_features)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features)
        )
        self.ln2 = nn.LayerNorm(n_features)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x, pad_mask=None, attn_mask=None):
        """
        :param x:            Input sequences.
        :type x:               torch.tensor of shape (max_seq_length, batch_size, n_features)
        :param pad_mask:     BoolTensor indicating which elements of the encoded source sequences should be ignored.
                             The values of True are ignored.
        :type pad_mask:        torch.BoolTensor of shape (batch_size, max_src_seq_length)
        :param attn_mask:    Subsequent mask to ignore subsequent elements of the target sequences in the inputs.
                             The rows of this matrix correspond to the output elements and the columns correspond
                             to the input elements.
        :type attn_mask:       torch.tensor of shape (max_tgt_seq_length, max_tgt_seq_length)
        :return:             Output tensor.
        :rtype:                torch.tensor of shape (max_seq_length, batch_size, n_features)
        Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
        """
        dx, _ = self.attn(query=x, key=x, value=x, key_padding_mask=pad_mask, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + self.dropout1(dx))

        dx = self.mlp(x)
        x = self.ln2(x + self.dropout2(dx))
        return x


class TrDecoderBlock(nn.Module):
    def __init__(self, n_features, n_attn_heads, n_hidden=64, dropout_prob=0.1):
        """
        :param n_features:   Number of input and output features.
        :param n_attn_heads: Number of attention heads in the Multi-Head Attention.
        :param n_hidden:     Number of hidden units in the Feedforward (MLP) block.
        :param dropout_prob: Dropout rate after the first layer of the MLP and in three places on the main path
                             (before combining the main path with a skip connection).
        """
        super(TrDecoderBlock, self).__init__()

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

    def forward(self, tgt_seq, src_seq, src_pad_mask=None, attn_mask=None):
        """
        :param tgt_seq:      Target sequences used as the inputs of the block.
        :type tgt_seq:         torch.tensor of shape (max_tgt_seq_length, batch_size, n_features)
        :param src_seq:      Encoded source sequences (outputs of the encoder).
        :type src_seq:         torch.tensor of shape (max_src_seq_length, batch_size, n_features)
        :param src_pad_mask: BoolTensor indicating which elements of the encoded source sequences should be ignored.
                             The values of True are ignored.
        :type src_pad_mask:    torch.BoolTensor of shape (batch_size, max_src_seq_length)
        :param attn_mask:    Subsequent mask to ignore subsequent elements of the target sequences in the inputs.
                             The rows of this matrix correspond to the output elements and the columns correspond
                             to the input elements.
        :type attn_mask:       torch.tensor of shape (max_tgt_seq_length, max_tgt_seq_length)
        :return:             Output tensor.
        :rtype:                torch.tensor of shape (max_seq_length, batch_size, n_features)
        Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
        """
        x = tgt_seq
        z = src_seq

        dx, _ = self.self_attn(query=x, key=x, value=x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + self.dropout1(dx))

        dx, _ = self.cross_attn(query=x, key=z, value=z, key_padding_mask=src_pad_mask, need_weights=False)
        x = self.ln2(x + self.dropout2(dx))

        dx = self.mlp(x)
        x = self.ln3(x + self.dropout3(dx))
        return x




