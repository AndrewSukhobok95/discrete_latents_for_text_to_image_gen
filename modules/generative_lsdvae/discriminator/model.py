import os
import torch
import torch.nn as nn

from modules.generative_lsdvae.blocks import DownSampleX2, EmbeddingMLP


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.eps = 1e-7
        ch_level_1 = 64
        ch_level_2 = 128
        ch_level_3 = 128

        ##############################
        ######## IMG ENCODING ########
        ##############################
        self.encoder_1 = nn.Sequential(
            DownSampleX2(in_channels=in_channels, out_channels=ch_level_1, bias=True, use_bn=True)
        )  # height: 28 -> 14

        self.encoder_2 = nn.Sequential(
            DownSampleX2(in_channels=ch_level_1, out_channels=ch_level_2, bias=True, use_bn=True)
        )  # height: 14 -> 7

        self.encoder_3 = nn.Sequential(
            DownSampleX2(in_channels=ch_level_2, out_channels=ch_level_3, bias=True, use_bn=True)
        )  # height: 7 -> 3

        self.uncond_discriminator = nn.Sequential(
            nn.Conv2d(ch_level_3, 1, kernel_size=3, stride=1, padding=0)
        )  # height: 3 -> 1

    def forward(self, img):
        img_feat_1 = self.encoder_1(img)
        img_feat_2 = self.encoder_2(img_feat_1)
        img_feat_3 = self.encoder_3(img_feat_2)
        cls_uncond = self.uncond_discriminator(img_feat_3).squeeze()
        return torch.sigmoid(cls_uncond)

    def save_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        path = os.path.join(root_path, model_name + "_uncond_discriminator.pth")
        torch.save(self.state_dict(), path)


# class Discriminator(nn.Module):
#     def __init__(self, in_channel, cond_in_dim, hidden_dim=128):
#         super(Discriminator, self).__init__()
#         self.eps = 1e-7
#         ch_level_1 = 64
#         ch_level_2 = 128
#         ch_level_3 = 128
#
#         ##############################
#         ######## IMG ENCODING ########
#         ##############################
#         self.encoder_1 = nn.Sequential(
#             DownSampleX2(in_channels=in_channel, out_channels=ch_level_1, bias=True, use_bn=True)
#         )  # height: 28 -> 14
#
#         self.encoder_2 = nn.Sequential(
#             DownSampleX2(in_channels=ch_level_1, out_channels=ch_level_2, bias=True, use_bn=True)
#         )  # height: 14 -> 7
#
#         self.encoder_3 = nn.Sequential(
#             DownSampleX2(in_channels=ch_level_2, out_channels=ch_level_3, bias=True, use_bn=True)
#         )  # height: 7 -> 3
#
#         self.uncond_discriminator = nn.Sequential(
#             nn.Conv2d(ch_level_3, 1, kernel_size=3, stride=1, padding=0)
#         )  # height: 3 -> 1
#
#         ##############################
#         ####### IMG VECTORIZER #######
#         ##############################
#         self.img_vectorizer_1 = nn.Sequential(
#             DownSampleX2(in_channels=ch_level_1, out_channels=hidden_dim, bias=True, use_bn=True),
#             nn.AvgPool2d(kernel_size=7))
#
#         self.img_vectorizer_2 = nn.Sequential(
#             DownSampleX2(in_channels=ch_level_2, out_channels=hidden_dim, bias=True, use_bn=True),
#             nn.AvgPool2d(kernel_size=3))
#
#         self.img_vectorizer_3 = nn.Sequential(
#             DownSampleX2(in_channels=ch_level_3, out_channels=hidden_dim, bias=True, use_bn=True),
#             nn.AvgPool2d(kernel_size=1))
#
#         ##############################
#         ######## COND PROCESS ########
#         ##############################
#         self.txt_encoder = nn.Sequential(
#             nn.Linear(cond_in_dim, hidden_dim),
#             nn.Tanh()
#         )
#
#         self.txt_weight_1 = nn.Sequential(
#             nn.Linear(cond_in_dim, hidden_dim + 1)
#         )
#
#         self.txt_weight_2 = nn.Sequential(
#             nn.Linear(cond_in_dim, hidden_dim + 1)
#         )
#
#         self.txt_weight_3 = nn.Sequential(
#             nn.Linear(cond_in_dim, hidden_dim + 1)
#         )
#
#         self.feat_level_weight = nn.Sequential(
#             nn.Linear(hidden_dim, 3),
#             nn.Softmax(-1)
#         )
#
#     def forward(self, img, condition):
#         img_feat_1 = self.encoder_1(img)
#         img_feat_2 = self.encoder_2(img_feat_1)
#         img_feat_3 = self.encoder_3(img_feat_2)
#
#         img_feats = [
#             self.img_vectorizer_1(img_feat_1),
#             self.img_vectorizer_2(img_feat_2),
#             self.img_vectorizer_3(img_feat_3)
#         ]
#
#         weights_cond = [
#             self.txt_weight_1(txt),
#             self.txt_weight_2(txt),
#             self.txt_weight_3(txt)
#         ]
#
#         txt_feat = self.txt_encoder(txt)
#         txt_attn = txt_feat.sum(-1).exp() * txt_mask
#         txt_attn = txt_attn / txt_attn.sum(0, keepdim=True)
#
#         cls_cond = 0
#
#         feat_level_weights = self.feat_level_weight(txt_feat).permute(2, 0, 1)
#
#         for i in range(len(img_feats)):
#             img_feat = img_feats[i].squeeze(-1)
#             W_cond = weights_cond[i]
#             W_cond, b_cond = W_cond[:, :, :-1], W_cond[:, :, [-1]]
#
#             cls_cond += torch.sigmoid(torch.bmm(W_cond, img_feat) + b_cond).squeeze(-1) * feat_level_weights[i, :, :]
#
#         cls_cond = torch.clamp(cls_cond + self.eps, min=0.0, max=1.0).pow(txt_attn).prod(dim=1)
#
#         if only_conditional:
#             return cls_cond
#
#         cls_uncond = self.uncond_discriminator(img_feat_3).squeeze()
#
#         return cls_uncond, cls_cond
#
#     def save_model(self, root_path, model_name):
#         if not os.path.exists(root_path):
#             os.makedirs(root_path)
#         path = os.path.join(root_path, model_name + "_discriminator.pth")
#         torch.save(self.state_dict(), path)


class cDiscriminator(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim=128,
                 dropout_prob=0.1):
        super(cDiscriminator, self).__init__()
        self.eps = 1e-7
        ch_level_1 = 64
        ch_level_2 = 128
        ch_level_3 = 128

        ##############################
        ######## IMG ENCODING ########
        ##############################
        self.encoder_1 = nn.Sequential(
            DownSampleX2(in_channels=in_channels, out_channels=ch_level_1, bias=True, use_bn=True)
        )  # height: 28 -> 14

        self.encoder_2 = nn.Sequential(
            DownSampleX2(in_channels=ch_level_1, out_channels=ch_level_2, bias=True, use_bn=True)
        )  # height: 14 -> 7

        self.encoder_3 = nn.Sequential(
            DownSampleX2(in_channels=ch_level_2, out_channels=ch_level_3, bias=True, use_bn=True)
        )  # height: 7 -> 3

        self.uncond_discriminator = nn.Sequential(
            nn.Conv2d(ch_level_3, 1, kernel_size=3, stride=1, padding=0)
        )  # height: 3 -> 1

        ##############################
        ####### IMG VECTORIZER #######
        ##############################
        self.img_vectorizer_1 = nn.Sequential(
            DownSampleX2(in_channels=ch_level_1, out_channels=hidden_dim, bias=True, use_bn=True),
            nn.AvgPool2d(kernel_size=7))

        self.img_vectorizer_2 = nn.Sequential(
            DownSampleX2(in_channels=ch_level_2, out_channels=hidden_dim, bias=True, use_bn=True),
            nn.AvgPool2d(kernel_size=3))

        self.img_vectorizer_3 = nn.Sequential(
            DownSampleX2(in_channels=ch_level_3, out_channels=hidden_dim, bias=True, use_bn=True),
            nn.AvgPool2d(kernel_size=1))

        ##############################
        ######## COND PROCESS ########
        ##############################

        self.cond_encoder = EmbeddingMLP(
            num_embeddings=10,
            embedding_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout_prob=dropout_prob)

        self.cond_weight_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim + 1),
            nn.ReLU(),
            nn.Linear(hidden_dim + 1, hidden_dim + 1)
        )

        self.cond_weight_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim + 1),
            nn.ReLU(),
            nn.Linear(hidden_dim + 1, hidden_dim + 1)
        )

        self.cond_weight_3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim + 1),
            nn.ReLU(),
            nn.Linear(hidden_dim + 1, hidden_dim + 1)
        )

    def forward(self, img, condition):
        img_feat_1 = self.encoder_1(img)
        img_feat_2 = self.encoder_2(img_feat_1)
        img_feat_3 = self.encoder_3(img_feat_2)

        cls_uncond = self.uncond_discriminator(img_feat_3).squeeze()

        img_feats = [
            self.img_vectorizer_1(img_feat_1),
            self.img_vectorizer_2(img_feat_2),
            self.img_vectorizer_3(img_feat_3)
        ]

        cond_feat = self.cond_encoder(condition)

        weights_cond = [
            self.cond_weight_1(cond_feat),
            self.cond_weight_2(cond_feat),
            self.cond_weight_3(cond_feat)
        ]

        return torch.sigmoid(cls_uncond)

    def save_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        path = os.path.join(root_path, model_name + "_uncond_discriminator.pth")
        torch.save(self.state_dict(), path)




