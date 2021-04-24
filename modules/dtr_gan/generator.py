import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.dtr_gan.blocks import PositionalEmbedding, MLP
from modules.common_blocks import TrEncoderBlock


class Generator(nn.Module):
    def __init__(self,
                 noise_dim,
                 hidden_width,
                 hidden_height,
                 embedding_dim,
                 num_blocks,
                 n_attn_heads,
                 hidden_dim,
                 dropout_prob,
                 num_latent_positions):
        super(Generator, self).__init__()

        self.hidden_width = hidden_width
        self.hidden_height = hidden_height
        self.embedding_dim = embedding_dim
        self.num_latent_positions = num_latent_positions

        self.noise_pe = PositionalEmbedding(
            embedding_dim=embedding_dim,
            num_embeddings=num_latent_positions)

        self.noise_mlp = MLP(
            in_dim=noise_dim,
            out_dim=embedding_dim * num_latent_positions,
            dropout_prob=dropout_prob)

        self.tr_encoder_blocks = nn.ModuleList([
            TrEncoderBlock(n_features=embedding_dim,
                           n_attn_heads=n_attn_heads,
                           n_hidden=hidden_dim,
                           dropout_prob=dropout_prob)
            for _ in range(num_blocks)
        ])

    def forward(self, noise, tau=1/16, hard=False):
        noise_codes = self.noise_mlp(noise)
        noise_codes = noise_codes.view(-1, self.num_latent_positions, self.embedding_dim)
        noise_codes = self.noise_pe(noise_codes)
        x = noise_codes.permute(1, 0, 2)

        for i, block in enumerate(self.tr_encoder_blocks):
            x = block(x)

        z_logits = x.permute(1, 2, 0).view(-1, self.embedding_dim, self.hidden_width, self.hidden_height)
        z = self.quantize(z_logits, tau=tau, hard=hard)
        return z

    def quantize(self, z_logits, tau=1/16, hard=False):
        return F.gumbel_softmax(z_logits, tau=tau, hard=hard, dim=1)


