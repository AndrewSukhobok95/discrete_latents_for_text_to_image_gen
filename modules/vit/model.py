import torch
import torch.nn.functional as F
from torch import nn, optim

from modules.common_blocks import TrEncoderBlock


class ViT(nn.Module):
    def __init__(self,
                 input_height,
                 input_width,
                 input_channels,
                 num_blocks,
                 hidden_dim,
                 n_attn_heads,
                 dropout_prob,
                 n_classes,
                 device=torch.device('cpu')):
        super(ViT, self).__init__()
        self.device = device

        num_positions = input_height * input_width + 1
        self.pe = nn.Parameter(torch.randn(num_positions, 1, input_channels))

        self.lin_proj = nn.Linear(input_channels, input_channels)

        self.cls_token = nn.Parameter(torch.randn(1, 1, input_channels))

        self.tr_encoder_blocks = nn.ModuleList([
            TrEncoderBlock(n_features=input_channels,
                           n_attn_heads=n_attn_heads,
                           n_hidden=hidden_dim,
                           dropout_prob=dropout_prob)
            for _ in range(num_blocks)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(input_channels),
            nn.Linear(input_channels, n_classes),
        )

        self.to(self.device)

    def forward(self, x, average_cls_token=False):
        seq_len, batch, emb = x.size()

        x = self.lin_proj(x)
        cls_tokens = self.cls_token.expand(-1, batch, -1)
        x = torch.cat((cls_tokens, x), dim=0)
        x += self.pe.repeat(1, batch, 1)

        for i, block in enumerate(self.tr_encoder_blocks):
            x = block(x)

        if average_cls_token:
            cls_input = x.mean(dim=0)
        else:
            cls_input = x[0, :, :]

        cls = self.mlp_head(cls_input).squeeze()
        return cls

