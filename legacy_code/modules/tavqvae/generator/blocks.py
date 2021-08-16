import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.common_blocks import Residual, ResidualStack, ChangeChannels


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(Attention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, imgh, texth, mask=None):
        """
        :param imgh: batch x emb_size x ih x iw (query_len=hxw)
        :param texth: seq_len x batch x emb_size
        :param mask: batch x seq_len
        """
        batch_size = imgh.size(0)
        ih, iw = imgh.size(2), imgh.size(3)
        query_len = ih * iw

        # --> batch x emb_size x query_len
        query = imgh.view(batch_size, -1, query_len)
        # --> query_len x batch x emb_size
        query = query.permute(2,0,1)

        # Mask in troch.multihead_attn notations
        # True value corresponds to padding places
        pad_mask = None
        if mask is not None:
            pad_mask = (1 - mask).bool()

        attn, _ = self.multihead_attn(query=query, key=texth, value=texth,
                                      key_padding_mask=pad_mask)

        # --> batch x emb_size x query_len
        attn = attn.permute(1,2,0)
        # --> batch x emb_size x ih x iw
        attn = attn.view(batch_size, -1, ih, iw)

        return attn


class MaskBlock(nn.Module):
    def __init__(self, ch_dim, bias=False):
        super(MaskBlock, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(in_channels=ch_dim, out_channels=ch_dim, kernel_size=1, stride=1, bias=bias),
            ResidualStack(ch_dim, ch_dim, 2, bias=bias, use_bn=False),
            ChangeChannels(in_channels=ch_dim, out_channels=1, bias=bias)
        )

    def forward(self, x):
        x = self._block(x)
        return F.hardsigmoid(x)


def masking_sum(x1, x2, mask):
    mask_ch = mask.repeat((1, x1.size(1), 1, 1))
    masked_x1 = x1 * mask_ch
    masked_x2 = x2 * (1 - mask_ch)
    return masked_x1 + masked_x2


def mask_tensor(x, mask):
    mask_ch = mask.repeat((1, x.size(1), 1, 1))
    return x * mask_ch



