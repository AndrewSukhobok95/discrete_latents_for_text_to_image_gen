import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from modules.generative_lsdvae.generator.model import Generator
from modules.generative_lsdvae.discriminator.model import Discriminator


class Config:
    DEVICE                = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_dim                   = 100
    hidden_height               = 7
    hidden_width                = 7
    vocab_size                  = 10
    img_channels                = 1
    num_tr_blocks               = 8
    dvae_num_x2upsamples        = 2
    dvae_num_resids_upsample    = 3
    dvae_num_resids_bottleneck  = 4
    dvae_hidden_dim             = 64
    tr_hidden_dim               = 64
    tr_n_attn_heads             = 10
    dropout_prob                = 0.1

    save_model_path = "/home/andrey/Aalto/thesis/TA-VQVAE/models/lsdvae/"
    save_model_name = "LSDVAE"
    load_dvae_path  = "/home/andrey/Aalto/thesis/TA-VQVAE/models/dvae_mnist"
    dvae_model_name = "dvae_mnist"

    NUM_EPOCHS                  = 80
    BATCH_SIZE                  = 512
    LR                          = 0.001
    LR_gamma                    = 0.35
    step_LR_milestones          = [5, 20, 60]
    temp_start                  = 1
    temp_end                    = 0.0625
    temp_steps                  = 100 * 20

CONFIG = Config()


trainset = datasets.MNIST(
    './data/MNIST/',
    train=True,
    download=False,
    transform=transforms.ToTensor())

train_loader = DataLoader(
    trainset,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=True)

G = Generator(
    noise_dim=CONFIG.noise_dim,
    hidden_height=CONFIG.hidden_height,
    hidden_width=CONFIG.hidden_width,
    hidden_vocab_size=CONFIG.vocab_size,
    out_channels=CONFIG.img_channels,
    num_tr_blocks=CONFIG.num_tr_blocks,
    dvae_num_x2upsamples=CONFIG.dvae_num_x2upsamples,
    dvae_num_resids_upsample=CONFIG.dvae_num_resids_upsample,
    dvae_num_resids_bottleneck=CONFIG.dvae_num_resids_bottleneck,
    dvae_hidden_dim=CONFIG.dvae_hidden_dim,
    tr_hidden_dim=CONFIG.tr_hidden_dim,
    tr_n_attn_heads=CONFIG.tr_n_attn_heads,
    dropout_prob=CONFIG.dropout_prob)
G.load_dvae_decoder(
    root_path=CONFIG.load_dvae_path,
    model_name=CONFIG.dvae_model_name)

D = Discriminator(
    in_channels=CONFIG.img_channels)

optimizer_G = optim.Adam(G.parameters_lat_generator(), lr=CONFIG.LR)
optimizer_D = optim.Adam(D.parameters(), lr=CONFIG.LR)

lr_scheduler_G = MultiStepLR(optimizer_G, milestones=CONFIG.step_LR_milestones, gamma=CONFIG.LR_gamma)
lr_scheduler_D = MultiStepLR(optimizer_D, milestones=CONFIG.step_LR_milestones, gamma=CONFIG.LR_gamma)


if __name__ == '__main__':
    print("Device in use: {}".format(CONFIG.DEVICE))

    iteration = 0
    for epoch in range(CONFIG.NUM_EPOCHS):
        for x, label in train_loader:
            label = label.to(CONFIG.DEVICE)
            real = x.to(CONFIG.DEVICE)

            current_batch_dim = real.sizes(0)
            labels_real = torch.full((current_batch_dim,), 1.0, device=CONFIG.DEVICE)
            labels_fake = torch.full((current_batch_dim,), 0.0, device=CONFIG.DEVICE)

            ############################
            ### Update Discriminator ###
            ############################
            D.zero_grad()
            noise = torch.randn(current_batch_dim, CONFIG.noise_dim, device=CONFIG.DEVICE)
            fake, z_logits, z = G(noise=noise, condition=label, hard=True)

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


            with torch.no_grad():
                n_used_codes = len(z.detach().cpu().argmax(dim=1).view(-1).unique())
                print("Epoch: {} Iter: {} GLoss: {} DLoss: {} Avg_D_real {} Avg_D_fake {} N codes used: {}".format(
                    epoch, iteration, g_loss.item(), d_loss.item(), avg_label_D_real, avg_label_D_fake, n_used_codes))

            iteration += 1




