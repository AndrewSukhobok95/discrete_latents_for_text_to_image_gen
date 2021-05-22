import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch import nn, optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import make_grid
from torchvision import transforms as torch_transforms

from modules.dvae.model import DVAE
from modules.seqGAN.generator.gen_noise import Generator
from datasets.mnist_loader import MNISTData
from modules.dvae.funcs import ng_quantize


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_channels = 1
    vocab_size = 32

    hidden_height = 7
    hidden_width = 7

    num_blocks = 8
    n_attn_heads = 8
    hidden_dim = 256
    dropout_prob = 0.1

    dvae_num_x2upsamples = 2
    dvae_num_resids_downsample = 3
    dvae_num_resids_bottleneck = 4
    dvae_hidden_dim = 256

    mnist_type = "classic"
    root_img_path = "/m/home/home8/82/sukhoba1/data/Desktop/TA-VQVAE/data/MNIST/"

    load_dvae_path = "/m/home/home8/82/sukhoba1/data/Desktop/TA-VQVAE/models/mnist/dvae_vocab32_mnist/"
    dvae_model_name = "dvae_vocab32_mnist"

    model_path = "/m/home/home8/82/sukhoba1/data/Desktop/TA-VQVAE/models/mnist/seq_GAN_v1/"
    model_name = "seq_GAN_v1"

    NUM_EPOCHS = 30
    BATCH_SIZE = 512
    LR = 0.001
    LR_gamma = 0.1
    step_LR_milestones = [5, 15, 25]


CONFIG = Config()

data_source = MNISTData(
    img_type=CONFIG.mnist_type,
    root_path=CONFIG.root_img_path,
    batch_size=CONFIG.BATCH_SIZE)
train_loader = data_source.get_train_loader()


dvae = DVAE(
    in_channels=CONFIG.img_channels,
    vocab_size=CONFIG.vocab_size,
    num_x2downsamples=CONFIG.dvae_num_x2upsamples,
    num_resids_downsample=CONFIG.dvae_num_resids_downsample,
    num_resids_bottleneck=CONFIG.dvae_num_resids_bottleneck,
    hidden_dim=CONFIG.dvae_hidden_dim)

G = Generator(
    hidden_width=CONFIG.hidden_width,
    hidden_height=CONFIG.hidden_height,
    embedding_dim=CONFIG.vocab_size,
    num_blocks=CONFIG.num_blocks,
    hidden_dim=CONFIG.hidden_dim,
    n_attn_heads=CONFIG.n_attn_heads,
    dropout_prob=CONFIG.dropout_prob)

optimizer_G = optim.Adam(G.parameters(), lr=CONFIG.LR)
















