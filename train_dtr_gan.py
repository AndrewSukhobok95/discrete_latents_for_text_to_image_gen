import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from modules.dtr_gan.generator import Generator
from modules.dtr_gan.discriminator import Discriminator
from modules.dvae.model import DVAE


class Config:
    DEVICE                      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_channels = 1
    vocab_size = 10

    noise_dim                   = 100
    hidden_height               = 7
    hidden_width                = 7

    num_blocks                  = 8
    n_attn_heads                = 10
    hidden_dim                  = 128
    dropout_prob                = 0.1

    dvae_num_x2upsamples        = 2
    dvae_num_resids_downsample  = 3
    dvae_num_resids_bottleneck  = 4
    dvae_hidden_dim             = 64


    save_model_path = "/home/andrey/Aalto/thesis/TA-VQVAE/models/lsdvae/"
    save_model_name = "LSDVAE"
    load_dvae_path  = "/home/andrey/Aalto/thesis/TA-VQVAE/models/dvae_mnist"
    dvae_model_name = "dvae_mnist"
    data_path       = "/home/andrey/Aalto/thesis/TA-VQVAE/data/MNIST/"

    NUM_EPOCHS                  = 100
    BATCH_SIZE                  = 512
    LR                          = 0.0001
    LR_gamma                    = 0.1
    step_LR_milestones          = [60]


CONFIG = Config()


trainset = datasets.MNIST(
    CONFIG.data_path,
    train=True,
    download=False,
    transform=transforms.ToTensor())

train_loader = DataLoader(
    trainset,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=True)

dvae = DVAE(
    in_channels=CONFIG.img_channels,
    vocab_size=CONFIG.vocab_size,
    num_x2downsamples=CONFIG.dvae_num_x2upsamples,
    num_resids_downsample=CONFIG.dvae_num_resids_downsample,
    num_resids_bottleneck=CONFIG.dvae_num_resids_bottleneck,
    hidden_dim=CONFIG.dvae_hidden_dim)

G = Generator(
    noise_dim=CONFIG.noise_dim,
    hidden_width=CONFIG.hidden_width,
    hidden_height=CONFIG.hidden_height,
    embedding_dim=CONFIG.vocab_size,
    num_blocks=CONFIG.num_blocks,
    n_attn_heads=CONFIG.n_attn_heads,
    hidden_dim=CONFIG.hidden_dim,
    dropout_prob=CONFIG.dropout_prob,
    num_latent_positions=CONFIG.hidden_width * CONFIG.hidden_height)

D = Discriminator(
    embedding_dim=CONFIG.vocab_size,
    num_blocks=CONFIG.num_blocks,
    n_attn_heads=CONFIG.n_attn_heads,
    hidden_dim=CONFIG.hidden_dim,
    dropout_prob=CONFIG.dropout_prob,
    num_latent_positions=CONFIG.hidden_width * CONFIG.hidden_height + 1)

optimizer_G = optim.Adam(G.parameters(), lr=CONFIG.LR)
optimizer_D = optim.Adam(D.parameters(), lr=CONFIG.LR)

lr_scheduler_G = MultiStepLR(optimizer_G, milestones=CONFIG.step_LR_milestones, gamma=CONFIG.LR_gamma)
lr_scheduler_D = MultiStepLR(optimizer_D, milestones=CONFIG.step_LR_milestones, gamma=CONFIG.LR_gamma)


if __name__ == '__main__':
    print("Device in use: {}".format(CONFIG.DEVICE))

    dvae.eval()
    G.train()
    D.train()

    dvae.load_model(
        root_path=CONFIG.load_dvae_path,
        model_name=CONFIG.dvae_model_name)

    dvae.to(CONFIG.DEVICE)
    G.to(CONFIG.DEVICE)
    D.to(CONFIG.DEVICE)

    iteration = 0
    for epoch in range(CONFIG.NUM_EPOCHS):
        for x, label in train_loader:
            x = x.to(CONFIG.DEVICE)
            with torch.no_grad():
                real = dvae.q_encode(x, hard=True)

            current_batch_dim = real.size(0)
            labels_real = torch.full((current_batch_dim,), 1.0, device=CONFIG.DEVICE)
            labels_fake = torch.full((current_batch_dim,), 0.0, device=CONFIG.DEVICE)

            ############################
            ### Update Discriminator ###
            ############################
            D.zero_grad()
            noise = torch.randn(current_batch_dim, CONFIG.noise_dim, device=CONFIG.DEVICE)
            fake = G(noise=noise, hard=True)

            labels_D_real = D(real)
            d_loss_real = F.binary_cross_entropy(labels_D_real, labels_real)

            labels_D_fake = D(fake.detach())
            d_loss_fake = F.binary_cross_entropy(labels_D_fake, labels_fake)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            avg_label_D_real = labels_D_real.mean().item()
            avg_label_D_fake = labels_D_fake.mean().item()

            ############################
            ##### Update Generator #####
            ############################
            G.zero_grad()

            labels_D_fake = D(fake)

            g_loss = F.binary_cross_entropy(labels_D_fake, labels_real)
            g_loss.backward()
            optimizer_G.step()


            # with torch.no_grad():
            #     n_used_codes = len(z.detach().cpu().argmax(dim=1).view(-1).unique())
            #     print("Epoch: {} Iter: {} GLoss: {} DLoss: {} Avg_D_real {} Avg_D_fake {} N codes used: {}".format(
            #         epoch, iteration, g_loss.item(), d_loss.item(), avg_label_D_real, avg_label_D_fake, n_used_codes))

            iteration += 1