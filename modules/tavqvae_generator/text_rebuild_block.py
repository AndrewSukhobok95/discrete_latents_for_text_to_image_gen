import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.tavqvae_generator.blocks import Attention, MaskBlock, masking_sum, mask_tensor
from modules.common_blocks import ChangeChannels, ResidualStack


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)


class TextRebuildBlock(nn.Module):
    def __init__(self, channel_dim, embed_dim, num_residual_layers, bias=False):
        super(TextRebuildBlock, self).__init__()
        self.convert_ch_1 = ChangeChannels(in_channels=channel_dim, out_channels=embed_dim, bias=bias)
        self.attn_1 = Attention(embed_dim=embed_dim)
        self.res_block_1 = ResidualStack(in_channels=embed_dim, out_channels=embed_dim,
                                         num_residual_layers=num_residual_layers, bias=bias)
        self.attn_2 = Attention(embed_dim=embed_dim)
        self.res_block_2 = ResidualStack(in_channels=embed_dim, out_channels=embed_dim,
                                         num_residual_layers=num_residual_layers, bias=bias)
        self.mask_block = MaskBlock(ch_dim=embed_dim, bias=bias)
        self.convert_ch_2 = ChangeChannels(in_channels=embed_dim, out_channels=channel_dim, bias=bias)

        self.apply(init_weights)

    def forward(self, imgh, texth, text_mask=None):
        """
        :param imgh: batch x ch x h x w (query_len=hxw)
        :param texth: seq_len x batch x emb_size
        :param text_mask: batch x seq_len
        """
        x = self.convert_ch_1(imgh)
        x = self.attn_1(x, texth, text_mask)
        x = self.res_block_1(x)
        x = self.attn_2(x, texth, text_mask)
        mask = self.mask_block(x)
        x = self.res_block_2(x)
        x = self.convert_ch_2(x)
        return imgh + mask_tensor(x, mask), mask



# class TextRebuildBlock(nn.Module):
#     def __init__(self, channel_dim, embed_dim, num_residual_layers, bias=False):
#         super(TextRebuildBlock, self).__init__()
#         self.convert_ch_1 = ChangeChannels(in_channels=channel_dim, out_channels=embed_dim, bias=bias)
#         self.attn_1 = Attention(embed_dim=embed_dim)
#         self.res_block = ResidualStack(in_channels=embed_dim, out_channels=embed_dim,
#                                        num_residual_layers=num_residual_layers, bias=bias)
#         self.attn_2 = Attention(embed_dim=embed_dim)
#         self.mask_block = MaskBlock(ch_dim=embed_dim, bias=bias)
#         self.convert_ch_2 = ChangeChannels(in_channels=embed_dim, out_channels=channel_dim, bias=bias)
#
#     def forward(self, imgh, texth, text_mask=None):
#         """
#         :param imgh: batch x ch x h x w (query_len=hxw)
#         :param texth: seq_len x batch x emb_size
#         :param text_mask: batch x seq_len
#         """
#         x = self.convert_ch_1(imgh)
#         x = self.attn_1(x, texth, text_mask)
#         x = self.res_block(x)
#         x = self.attn_2(x, texth, text_mask)
#         mask = self.mask_block(x)
#         x = self.convert_ch_2(x)
#         return masking_sum(x, imgh, mask), mask


