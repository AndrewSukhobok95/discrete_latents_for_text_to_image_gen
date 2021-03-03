import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=bias)
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


class Attention(nn.Module):
    def __init__(self, embed_dim, text_embed_dim, num_heads=10):
        super(Attention, self).__init__()
        self.multihead_attn_1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.embed_proj = nn.Linear(text_embed_dim, embed_dim)

    def forward(self, imgh, texth, mask=None):
        """
        :param imgh: batch x ch x ih x iw (query_len=hxw)
        :param texth: batch x seq_len x emb_size
        """
        batch_size = imgh.size(0)
        ih, iw = imgh.size(2), imgh.size(3)
        query_len = ih * iw

        # --> batch x ch x query_len
        query = imgh.view(batch_size, -1, query_len)
        query = self.embed_proj(query)

        attn_output, _ = self.multihead_attn_1(
            query=query, key=texth, value=texth, key_padding_mask=mask)

        # targetT = torch.transpose(target, 1, 2).contiguous()
        # # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        # sourceT = context.unsqueeze(3)
        # # --> batch x idf x sourceL
        # sourceT = self.conv_context(sourceT).squeeze(3)
        #
        # # Get attention
        # # (batch x queryL x idf)(batch x idf x sourceL)
        # # -->batch x queryL x sourceL
        # attn = torch.bmm(targetT, sourceT)
        # # --> batch*queryL x sourceL
        # attn = attn.view(batch_size*queryL, sourceL)
        # if self.mask is not None:
        #     # batch_size x sourceL --> batch_size*queryL x sourceL
        #     mask = self.mask.repeat(queryL, 1)
        #     attn.data.masked_fill_(mask.data, -float('inf'))
        # attn = self.sm(attn)  # Eq. (2)
        # # --> batch x queryL x sourceL
        # attn = attn.view(batch_size, queryL, sourceL)
        # # --> batch x sourceL x queryL
        # attn = torch.transpose(attn, 1, 2).contiguous()
        #
        # # (batch x idf x sourceL)(batch x sourceL x queryL)
        # # --> batch x idf x queryL
        # weightedContext = torch.bmm(sourceT, attn)
        # weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        # attn = attn.view(batch_size, -1, ih, iw)

        return



