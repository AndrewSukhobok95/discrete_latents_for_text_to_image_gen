import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.tavqvae.blocks import Attention, MaskBlock, masking_sum
from modules.common_blocks import ChangeChannels, ResidualStack


class TextRebuildBlock(nn.Module):
    def __init__(self, channel_dim, embed_dim, num_residual_layers, bias=False):
        super(TextRebuildBlock, self).__init__()
        self.convert_ch_1 = ChangeChannels(in_channels=channel_dim, out_channels=embed_dim, bias=bias)
        self.attn_1 = Attention(embed_dim=embed_dim)
        self.res_block = ResidualStack(in_channels=embed_dim, out_channels=embed_dim,
                                       num_residual_layers=num_residual_layers, bias=bias)
        self.attn_2 = Attention(embed_dim=embed_dim)
        self.mask_block = MaskBlock(ch_dim=embed_dim, bias=bias)
        self.convert_ch_2 = ChangeChannels(in_channels=embed_dim, out_channels=channel_dim, bias=bias)

    def forward(self, imgh, texth, text_mask=None):
        """
        :param imgh: batch x ch x h x w (query_len=hxw)
        :param texth: seq_len x batch x emb_size
        :param text_mask: batch x seq_len
        """
        x = self.convert_ch_1(imgh)
        x = self.attn_1(x, texth, text_mask)
        x = self.res_block(x)
        x = self.attn_2(x, texth, text_mask)
        mask = self.mask_block(x)
        x = self.convert_ch_2(x)
        return masking_sum(x, imgh, mask), mask


