import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from einops.layers.torch import Rearrange
from modules.common_blocks import TrEncoderBlock


# class ViT(nn.Module):
#     def __init__(self,
#                  input_height,
#                  input_width,
#                  input_channels,
#                  num_blocks,
#                  hidden_dim,
#                  n_attn_heads,
#                  dropout_prob,
#                  n_classes,
#                  sigmoid_output=False,
#                  device=torch.device('cpu')):
#         super(ViT, self).__init__()
#         self.device = device
#         self.sigmoid_output = sigmoid_output
#
#         num_positions = input_height * input_width + 1
#         self.pe = nn.Parameter(torch.randn(num_positions, 1, input_channels))
#
#         self.lin_proj = nn.Linear(input_channels, input_channels)
#
#         self.cls_token = nn.Parameter(torch.randn(1, 1, input_channels))
#
#         self.tr_encoder_blocks = nn.ModuleList([
#             TrEncoderBlock(n_features=input_channels,
#                            n_attn_heads=n_attn_heads,
#                            n_hidden=hidden_dim,
#                            dropout_prob=dropout_prob,
#                            norm_first=True)
#             for _ in range(num_blocks)
#         ])
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(input_channels),
#             nn.Linear(input_channels, n_classes),
#         )
#
#         self.to(self.device)
#
#     def forward(self, x, average_cls_token=False):
#         seq_len, batch, emb = x.size()
#
#         x = self.lin_proj(x)
#         cls_tokens = self.cls_token.expand(-1, batch, -1)
#         x = torch.cat((cls_tokens, x), dim=0)
#         x += self.pe.repeat(1, batch, 1)
#
#         for i, block in enumerate(self.tr_encoder_blocks):
#             x = block(x)
#
#         if average_cls_token:
#             cls_input = x.mean(dim=0)
#         else:
#             cls_input = x[0, :, :]
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


class ViT(nn.Module):
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
                 out_dim,
                 sigmoid_output=False,
                 device=torch.device('cpu')):
        super(ViT, self).__init__()
        self.device = device
        self.sigmoid_output = sigmoid_output

        self.n_h_patch = img_height // patch_height
        self.n_w_patch = img_width // patch_width
        patch_dim = img_channels * patch_height * patch_width

        self.img_pe_col = nn.Parameter(torch.randn(self.n_h_patch, 1, embed_dim))
        self.img_pe_row = nn.Parameter(torch.randn(self.n_w_patch, 1, embed_dim))

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, embed_dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.tr_encoder_blocks = nn.ModuleList([
            TrEncoderBlock(n_features=embed_dim,
                           n_attn_heads=n_attn_heads,
                           n_hidden=hidden_dim,
                           dropout_prob=dropout_prob,
                           norm_first=True)
            for _ in range(num_blocks)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

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
            cls_input = full_x.mean(dim=0)
        else:
            cls_input = full_x[0, :, :]

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


