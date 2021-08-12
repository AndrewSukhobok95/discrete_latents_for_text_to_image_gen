import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from modules.clip.blocks import ImgEncoder, DVAEImgEncoder, TxtEncoder


class CLIP(nn.Module):
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
                 device=torch.device('cpu')):
        super(CLIP, self).__init__()
        self.device = device

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.img_model = ImgEncoder(
            img_height=img_height,
            img_width=img_width,
            img_channels=img_channels,
            patch_height=patch_height,
            patch_width=patch_width,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            n_attn_heads=n_attn_heads,
            dropout_prob=dropout_prob,
            device=device
        )

        self.txt_model = TxtEncoder(
            txt_max_length=txt_max_length,
            txt_vocab_size=txt_vocab_size,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            n_attn_heads=n_attn_heads,
            dropout_prob=dropout_prob,
            device=device
        )

        self.multimodal_embedding = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False)
        )

        self.to(device)

    def forward(self, x_img, x_txt):
        image_features = self.img_model(x_img)
        text_features = self.txt_model(x_txt)

        image_features = self.multimodal_embedding(image_features)
        text_features = self.multimodal_embedding(text_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text

    def save_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        path = os.path.join(root_path, model_name + ".pth")
        torch.save(self.state_dict(), path)

    def load_model(self, root_path, model_name, map_location=torch.device('cpu')):
        path = os.path.join(root_path, model_name + ".pth")
        self.load_state_dict(torch.load(path, map_location=map_location))


class DVAECLIP(nn.Module):
    def __init__(self,
                 img_latent_height,
                 img_latent_width,
                 img_latent_channels,
                 txt_max_length,
                 txt_vocab_size,
                 embed_dim,
                 num_blocks,
                 hidden_dim,
                 n_attn_heads,
                 dropout_prob,
                 device=torch.device('cpu')):
        super(DVAECLIP, self).__init__()
        self.device = device

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.img_model = DVAEImgEncoder(
            latent_height=img_latent_height,
            latent_width=img_latent_width,
            latent_channels=img_latent_channels,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            n_attn_heads=n_attn_heads,
            dropout_prob=dropout_prob,
            device=device
        )

        self.txt_model = TxtEncoder(
            txt_max_length=txt_max_length,
            txt_vocab_size=txt_vocab_size,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            n_attn_heads=n_attn_heads,
            dropout_prob=dropout_prob,
            device=device
        )

        self.multimodal_embedding = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False)
        )

        self.to(device)

    def forward(self, x_img, x_txt):
        image_features = self.img_model(x_img)
        text_features = self.txt_model(x_txt)

        image_features = self.multimodal_embedding(image_features)
        text_features = self.multimodal_embedding(text_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text

    def save_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        path = os.path.join(root_path, model_name + ".pth")
        torch.save(self.state_dict(), path)

    def load_model(self, root_path, model_name, map_location=torch.device('cpu')):
        path = os.path.join(root_path, model_name + ".pth")
        self.load_state_dict(torch.load(path, map_location=map_location))



