import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from einops.layers.torch import Rearrange
from modules.common_blocks import TrEncoderBlock


class TrMatcher(nn.Module):
    def __init__(self,
                 img_height,
                 img_width,
                 img_channels,
                 patch_height,
                 patch_width,
                 txt_max_length,
                 txt_vocab_size,
                 embed_dim,
                 num_blocks,
                 hidden_dim,
                 n_attn_heads,
                 dropout_prob,
                 tr_norm_first,
                 out_dim,
                 sigmoid_output=False,
                 device=torch.device('cpu')):
        super(TrMatcher, self).__init__()
        self.device = device
        self.sigmoid_output = sigmoid_output

        self.n_h_patch = img_height // patch_height
        self.n_w_patch = img_width // patch_width
        patch_dim = img_channels * patch_height * patch_width

        self.img_pe_col = nn.Parameter(torch.randn(self.n_h_patch, 1, embed_dim))
        self.img_pe_row = nn.Parameter(torch.randn(self.n_w_patch, 1, embed_dim))
        self.txt_pe = nn.Parameter(torch.randn(txt_max_length, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.text_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=txt_vocab_size, embedding_dim=embed_dim)
        )

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, embed_dim),
        )

        self.tr_encoder_blocks = nn.ModuleList([
            TrEncoderBlock(n_features=embed_dim,
                           n_attn_heads=n_attn_heads,
                           n_hidden=hidden_dim,
                           dropout_prob=dropout_prob,
                           norm_first=tr_norm_first)
            for _ in range(num_blocks)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self.to(self.device)

    def forward(self, img, txt_tokens):
        batch, ch, h, w = img.size()

        x = self.to_patch_embedding(img)
        x = x.permute(1, 0, 2)

        pe_column = self.img_pe_col.repeat(self.n_w_patch, 1, 1)
        pe_row = self.img_pe_row.repeat_interleave(self.n_h_patch, dim=0)
        x = x + pe_column + pe_row

        t = self.text_encoder(txt_tokens)
        t = t + self.txt_pe.repeat(1, batch, 1)

        cls_tokens = self.cls_token.expand(-1, batch, -1)

        full_x = torch.cat([cls_tokens, t, x], dim=0)

        for i, block in enumerate(self.tr_encoder_blocks):
            full_x = block(full_x)

        cls_input = x[0, :, :]

        cls = self.mlp_head(cls_input).squeeze()

        if self.sigmoid_output:
            return torch.sigmoid(cls)
        return cls

    def save_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        path = os.path.join(root_path, model_name + ".pth")
        torch.save(self.state_dict(), path)

    def load_model(self, root_path, model_name, map_location=torch.device('cpu')):
        path = os.path.join(root_path, model_name + ".pth")
        self.load_state_dict(torch.load(path, map_location=map_location))





# class TrMatcher(nn.Module):
#     def __init__(self,
#                  img_height,
#                  img_width,
#                  img_embed_dim,
#                  txt_max_length,
#                  txt_vocab_size,
#                  embed_dim,
#                  num_blocks,
#                  hidden_dim,
#                  n_attn_heads,
#                  dropout_prob,
#                  out_dim,
#                  sigmoid_output=False,
#                  device=torch.device('cpu')):
#         super(TrMatcher, self).__init__()
#         self.device = device
#         self.sigmoid_output = sigmoid_output
#         self.img_height = img_height
#         self.img_width = img_width
#
#         self.img_pe_col = nn.Parameter(torch.randn(img_height, 1, embed_dim))
#         self.img_pe_row = nn.Parameter(torch.randn(img_width, 1, embed_dim))
#         self.txt_pe = nn.Parameter(torch.randn(txt_max_length, 1, embed_dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#
#         self.text_encoder = nn.Sequential(
#             nn.Embedding(num_embeddings=txt_vocab_size, embedding_dim=embed_dim)
#         )
#
#         self.img_lin_proj = nn.Linear(img_embed_dim, embed_dim)
#
#         self.tr_encoder_blocks = nn.ModuleList([
#             TrEncoderBlock(n_features=embed_dim,
#                            n_attn_heads=n_attn_heads,
#                            n_hidden=hidden_dim,
#                            dropout_prob=dropout_prob)
#             for _ in range(num_blocks)
#         ])
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(embed_dim),
#             nn.Linear(embed_dim, out_dim),
#         )
#
#         self.to(self.device)
#
#     def forward(self, img, txt_tokens):
#
#         seq_len, batch, emb = img.size()
#         x = self.img_lin_proj(img)
#
#         pe_column = self.img_pe_col.repeat(self.img_width, 1, 1)
#         pe_row = self.img_pe_row.repeat_interleave(self.img_height, dim=0)
#         x = x + pe_column + pe_row
#
#         t = self.text_encoder(txt_tokens)
#         t = t + self.txt_pe.repeat(1, batch, 1)
#
#         cls_tokens = self.cls_token.expand(-1, batch, -1)
#
#         full_x = torch.cat([cls_tokens, t, x], dim=0)
#
#         for i, block in enumerate(self.tr_encoder_blocks):
#             full_x = block(full_x)
#
#         cls_input = x[0, :, :]
#
#         cls = self.mlp_head(cls_input).squeeze()
#
#         if self.sigmoid_output:
#             return torch.sigmoid(cls)
#         return cls
#
#     def save_model(self, root_path, model_name):
#         if not os.path.exists(root_path):
#             os.makedirs(root_path)
#         path = os.path.join(root_path, model_name + ".pth")
#         torch.save(self.state_dict(), path)
#
#     def load_model(self, root_path, model_name, map_location=torch.device('cpu')):
#         path = os.path.join(root_path, model_name + ".pth")
#         self.load_state_dict(torch.load(path, map_location=map_location))
