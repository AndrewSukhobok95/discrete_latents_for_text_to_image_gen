import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.dtr_gan.blocks import PositionalEmbedding
from modules.common_blocks import TrEncoderBlock


class Discriminator(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_blocks,
                 n_attn_heads,
                 hidden_dim,
                 dropout_prob,
                 num_latent_positions):
        super(Discriminator, self).__init__()

        self.pe = PositionalEmbedding(embedding_dim, num_embeddings=num_latent_positions)
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
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img_latent):
        b, c, h, w = img_latent.size()
        x = img_latent.view(b, c, h * w).permute(0, 2, 1)  # -> b, h*w, c

        x = self.lin_proj(x)
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pe(x)

        x = x.permute(1, 0, 2)

        for i, block in enumerate(self.tr_encoder_blocks):
            x = block(x)

        cls = self.mlp_head(x[0, :, :]).squeeze()

        return cls




