import os
import torch
import torch.nn as nn
from modules.tadvae.discriminator.blocks import DownSampleX2


class Discriminator(nn.Module):
    def __init__(self, txt_in_dim, hidden_dim=512):
        super(Discriminator, self).__init__()
        self.eps = 1e-7

        self.encoder_1 = nn.Sequential(
            DownSampleX2(in_channels=3, out_channels=32, bias=True, use_bn=True),
            DownSampleX2(in_channels=32, out_channels=64, bias=True, use_bn=True),
            DownSampleX2(in_channels=64, out_channels=128, bias=True, use_bn=True)
        )  # height: 256 -> 32

        self.encoder_2 = nn.Sequential(
            DownSampleX2(in_channels=128, out_channels=256, bias=True, use_bn=True),
            DownSampleX2(in_channels=256, out_channels=512, bias=True, use_bn=True)
        )  # height: 32 -> 8

        self.encoder_3 = nn.Sequential(
            DownSampleX2(in_channels=512, out_channels=512, bias=True, use_bn=True)
        )  # height: 8 -> 4

        self.uncond_discriminator = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        )  # height: 4 -> 1

        self.img_vectorizer_1 = nn.Sequential(
            DownSampleX2(in_channels=128, out_channels=hidden_dim, bias=True, use_bn=True),
            nn.AvgPool2d(kernel_size=16)
        )

        self.img_vectorizer_2 = nn.Sequential(
            DownSampleX2(in_channels=512, out_channels=hidden_dim, bias=True, use_bn=True),
            nn.AvgPool2d(kernel_size=4)
        )

        self.img_vectorizer_3 = nn.Sequential(
            DownSampleX2(in_channels=512, out_channels=hidden_dim, bias=True, use_bn=True),
            nn.AvgPool2d(kernel_size=2)
        )

        self.txt_encoder = nn.Sequential(
            nn.Linear(txt_in_dim, hidden_dim),
            nn.Tanh()
        )

        self.txt_weight_1 = nn.Sequential(
            nn.Linear(txt_in_dim, hidden_dim + 1)
        )

        self.txt_weight_2 = nn.Sequential(
            nn.Linear(txt_in_dim, hidden_dim + 1)
        )

        self.txt_weight_3 = nn.Sequential(
            nn.Linear(txt_in_dim, hidden_dim + 1)
        )

        self.feat_level_weight = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Softmax(-1)
        )

    def forward(self, img, txt, txt_mask, only_conditional=False):
        img_feat_1 = self.encoder_1(img)
        img_feat_2 = self.encoder_2(img_feat_1)
        img_feat_3 = self.encoder_3(img_feat_2)

        img_feats = [
            self.img_vectorizer_1(img_feat_1),
            self.img_vectorizer_2(img_feat_2),
            self.img_vectorizer_3(img_feat_3)
        ]

        weights_cond = [
            self.txt_weight_1(txt),
            self.txt_weight_2(txt),
            self.txt_weight_3(txt)
        ]

        txt_feat = self.txt_encoder(txt)
        txt_attn = txt_feat.sum(-1).exp() * txt_mask
        txt_attn = txt_attn / txt_attn.sum(0, keepdim=True)

        cls_cond = 0

        feat_level_weights = self.feat_level_weight(txt_feat).permute(2, 0, 1)

        for i in range(len(img_feats)):
            img_feat = img_feats[i].squeeze(-1)
            W_cond = weights_cond[i]
            W_cond, b_cond = W_cond[:, :, :-1], W_cond[:, :, [-1]]

            cls_cond += torch.sigmoid(torch.bmm(W_cond, img_feat) + b_cond).squeeze(-1) * feat_level_weights[i, :, :]

        cls_cond = torch.clamp(cls_cond + self.eps, min=0.0, max=1.0).pow(txt_attn).prod(dim=1)

        if only_conditional:
            return cls_cond

        cls_uncond = self.uncond_discriminator(img_feat_3).squeeze()

        return cls_uncond, cls_cond

    def save_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        path = os.path.join(root_path, model_name + "_discriminator.pth")
        torch.save(self.state_dict(), path)


