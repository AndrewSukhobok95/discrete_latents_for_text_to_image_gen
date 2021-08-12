import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from einops.layers.torch import Rearrange
from modules.common_blocks import TrEncoderBlock
from modules.dvae.model import DVAE


class ImgEncoder(nn.Module):
    def __init__(self,
                 img_height,
                 img_width,
                 img_channels,
                 patch_height,
                 patch_width,
                 embed_dim,
                 num_blocks,
                 hidden_dim,
                 n_attn_heads,
                 dropout_prob,
                 device=torch.device('cpu')):
        super(ImgEncoder, self).__init__()
        self.device = device

        self.n_h_patch = img_height // patch_height
        self.n_w_patch = img_width // patch_width
        patch_dim = img_channels * patch_height * patch_width

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.img_pe_col = nn.Parameter(torch.randn(self.n_h_patch, 1, embed_dim))
        self.img_pe_row = nn.Parameter(torch.randn(self.n_w_patch, 1, embed_dim))

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, embed_dim),
        )

        self.tr_encoder_blocks = nn.ModuleList([
            TrEncoderBlock(n_features=embed_dim,
                           n_attn_heads=n_attn_heads,
                           n_hidden=hidden_dim,
                           dropout_prob=dropout_prob,
                           norm_first=True)
            for _ in range(num_blocks)
        ])

        self.to(self.device)

    def forward(self, x, average_cls_token=False):
        batch, ch, h, w = x.size()

        x = self.to_patch_embedding(x)
        x = x.permute(1, 0, 2)

        pe_column = self.img_pe_col.repeat(self.n_w_patch, batch, 1)
        pe_row = self.img_pe_row.repeat_interleave(self.n_h_patch, dim=0).repeat(1, batch, 1)
        x = x + pe_column + pe_row

        cls_tokens = self.cls_token.expand(-1, batch, -1)

        full_x = torch.cat([cls_tokens, x], dim=0)
        for i, block in enumerate(self.tr_encoder_blocks):
            full_x = block(full_x)

        if average_cls_token:
            cls_out_token = full_x.mean(dim=0)
        else:
            cls_out_token = full_x[0, :, :]

        return cls_out_token


class DVAEImgEncoder(nn.Module):
    def __init__(self,
                 latent_height,
                 latent_width,
                 latent_channels,
                 embed_dim,
                 num_blocks,
                 hidden_dim,
                 n_attn_heads,
                 dropout_prob,
                 device=torch.device('cpu')):
        super(DVAEImgEncoder, self).__init__()
        self.device = device
        self.latent_height = latent_height
        self.latent_width = latent_width

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.img_pe_col = nn.Parameter(torch.randn(self.latent_height, 1, embed_dim))
        self.img_pe_row = nn.Parameter(torch.randn(self.latent_width, 1, embed_dim))

        self.lin_proj = nn.Linear(latent_channels, embed_dim)

        self.tr_encoder_blocks = nn.ModuleList([
            TrEncoderBlock(n_features=embed_dim,
                           n_attn_heads=n_attn_heads,
                           n_hidden=hidden_dim,
                           dropout_prob=dropout_prob,
                           norm_first=True)
            for _ in range(num_blocks)
        ])

        self.to(self.device)

    def forward(self, x, average_cls_token=False):
        seq_len, batch, emb = x.size()

        cls_tokens = self.cls_token.expand(-1, batch, -1)

        x = self.lin_proj(x)
        pe_column = self.img_pe_col.repeat(self.n_w_patch, batch, 1)
        pe_row = self.img_pe_row.repeat_interleave(self.n_h_patch, dim=0).repeat(1, batch, 1)
        x = x + pe_column + pe_row

        full_x = torch.cat([cls_tokens, x], dim=0)
        for i, block in enumerate(self.tr_encoder_blocks):
            full_x = block(full_x)

        if average_cls_token:
            cls_out_token = full_x.mean(dim=0)
        else:
            cls_out_token = full_x[0, :, :]

        return cls_out_token


class TxtEncoder(nn.Module):
    def __init__(self,
                 txt_max_length,
                 txt_vocab_size,
                 embed_dim,
                 num_blocks,
                 hidden_dim,
                 n_attn_heads,
                 dropout_prob,
                 device=torch.device('cpu')):
        super(TxtEncoder, self).__init__()
        self.device = device

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.txt_pe = nn.Parameter(torch.randn(txt_max_length, 1, embed_dim))

        self.txt_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=txt_vocab_size, embedding_dim=embed_dim)
        )

        self.tr_encoder_blocks = nn.ModuleList([
            TrEncoderBlock(n_features=embed_dim,
                           n_attn_heads=n_attn_heads,
                           n_hidden=hidden_dim,
                           dropout_prob=dropout_prob,
                           norm_first=True)
            for _ in range(num_blocks)
        ])

        self.to(self.device)

    def forward(self, x, average_cls_token=False):
        seq_len, batch = x.size()

        t = self.txt_embedding(x)
        t = t + self.txt_pe.repeat(1, batch, 1)

        cls_tokens = self.cls_token.expand(-1, batch, -1)

        full_x = torch.cat([cls_tokens, t], dim=0)
        for i, block in enumerate(self.tr_encoder_blocks):
            full_x = block(full_x)

        if average_cls_token:
            cls_out_token = full_x.mean(dim=0)
        else:
            cls_out_token = full_x[0, :, :]

        return cls_out_token




