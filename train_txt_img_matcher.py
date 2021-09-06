import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from modules.dvae.model import DVAE
from config_reader import ConfigReader
from datasets.mnist_loader import MNISTData
from utilities.md_mnist_utils import LabelsInfo
from modules.matcher.model import TrMatcher


config_dir = '/home/andrey/Aalto/thesis/TA-VQVAE/configs/finished/'
# config_dir = '/u/82/sukhoba1/unix/Desktop/TA-VQVAE/configs/finished/'
config_path = config_dir + 'Tr1s2s_mnistmd_v256_ds2_nb12.yaml'
CONFIG = ConfigReader(config_path=config_path)

CONFIG.BATCH_SIZE = 4


data_source = MNISTData(
    img_type=CONFIG.dataset_type,
    root_path="/home/andrey/Aalto/thesis/TA-VQVAE/data/multi_descriptive_MNIST/",#CONFIG.root_path,
    batch_size=CONFIG.BATCH_SIZE)

train_loader = data_source.get_train_loader(batch_size=CONFIG.BATCH_SIZE)


dvae = DVAE(
    in_channels=CONFIG.in_channels,
    vocab_size=CONFIG.vocab_size,
    num_x2downsamples=CONFIG.num_x2downsamples,
    num_resids_downsample=CONFIG.num_resids_downsample,
    num_resids_bottleneck=CONFIG.num_resids_bottleneck,
    hidden_dim=CONFIG.hidden_dim,
    device=CONFIG.DEVICE)

dvae.eval()
dvae.load_model(
    root_path="/home/andrey/Aalto/thesis/TA-VQVAE/models/mnist_md/dvae_v256_ds2/",#CONFIG.vae_model_path,
    model_name=CONFIG.vae_model_name)

model = TrMatcher(
    img_height=128,
    img_width=128,
    img_channels=3,
    patch_height=8,
    patch_width=8,
    txt_max_length=12,
    txt_vocab_size=20,
    embed_dim=128,
    num_blocks=8,
    hidden_dim=512,
    n_attn_heads=8,
    dropout_prob=0.1,
    tr_norm_first=True,
    out_dim=1,
    sigmoid_output=True)

# model = TrMatcher(
#     img_height=CONFIG.hidden_height,
#     img_width=CONFIG.hidden_width,
#     img_embed_dim=256,
#     txt_max_length=CONFIG.cond_seq_size,
#     txt_vocab_size=CONFIG.cond_vocab_size,
#     embed_dim=CONFIG.vocab_size,
#     num_blocks=8,
#     hidden_dim=512,
#     n_attn_heads=8,
#     dropout_prob=0.1,
#     out_dim=1,
#     sigmoid_output=True)

model.train()

optimizer = optim.Adam(model.parameters(), lr=CONFIG.LR)

if __name__=="__main__":
    iteration = 0
    for epoch in range(CONFIG.NUM_EPOCHS):
        for batch_index, (img, txt) in enumerate(train_loader):
            n_true = CONFIG.BATCH_SIZE // 2
            true_txt = txt[:n_true, :]
            false_txt = txt[n_true:, :]
            false_txt = torch.cat((false_txt[[-1], :], false_txt[:-1, :]), dim=0)
            txt = torch.cat((true_txt, false_txt), dim=0)

            match_labels = torch.zeros(CONFIG.BATCH_SIZE)
            match_labels[:n_true] = 1.0

            img = img.to(CONFIG.DEVICE)
            txt = txt.permute(1, 0).to(CONFIG.DEVICE)
            match_labels = match_labels.to(CONFIG.DEVICE)

            # with torch.no_grad():
            #     latent = dvae.ng_q_encode(img)
            #
            # b, emb, h, w = latent.size()
            # x = latent.view(b, emb, -1).permute(2, 0, 1)

            pred_labels = model(img, txt)

            loss = F.binary_cross_entropy(pred_labels, match_labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()



