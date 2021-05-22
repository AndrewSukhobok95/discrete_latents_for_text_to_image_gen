import os
import torch
from torch import nn
import torch.nn.functional as F
from modules.common_blocks import TrEncoderBlock


class Discriminator(nn.Module):
    def __init__(self,
                 hidden_height,
                 hidden_width,
                 embedding_dim,
                 num_blocks,
                 hidden_dim,
                 n_attn_heads,
                 dropout_prob):
        super(Discriminator, self).__init__()
        n_classes = 1

        num_latent_positions = hidden_height * hidden_width + 1
        self.pe = nn.Parameter(torch.randn(num_latent_positions, 1, embedding_dim))

        self.lin_proj = nn.Linear(embedding_dim, embedding_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.tr_encoder_blocks = nn.ModuleList([
            TrEncoderBlock(n_features=embedding_dim,
                           n_attn_heads=n_attn_heads,
                           n_hidden=hidden_dim,
                           dropout_prob=dropout_prob)
            for _ in range(num_blocks)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, n_classes),
        )

    def forward(self, x):
        seq_len, batch, emb = x.size()

        x = self.lin_proj(x)
        cls_tokens = self.cls_token.expand(-1, batch, -1)
        x = torch.cat((cls_tokens, x), dim=0)
        x += self.pe.repeat(1, batch, 1)

        for i, block in enumerate(self.tr_encoder_blocks):
            x = block(x)

        # cls_input = x.mean(dim=0)
        cls_input = x[0, :, :]
        cls = self.mlp_head(cls_input).squeeze()

        return torch.sigmoid(cls)


