import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from modules.dvae.model import DVAE
from config_reader import ConfigReader
from datasets.mnist_loader import MNISTData
from utilities.md_mnist_utils import LabelsInfo


import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from modules.common_blocks import TrEncoderBlock


config_dir = '/home/andrey/Aalto/thesis/TA-VQVAE/configs/finished/'
# config_dir = '/u/82/sukhoba1/unix/Desktop/TA-VQVAE/configs/finished/'
config_path = config_dir + 'trArC1s2s_mnistmd_v256_ds2_nb12_remote.yaml'
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


class TrMatcher(nn.Module):
    def __init__(self,
                 img_height,
                 img_width,
                 embed_dim,
                 txt_max_length,
                 txt_vocab_size,
                 num_blocks,
                 hidden_dim,
                 n_attn_heads,
                 dropout_prob,
                 out_dim,
                 sigmoid_output=False,
                 device=torch.device('cpu')):
        super(TrMatcher, self).__init__()
        self.device = device
        self.sigmoid_output = sigmoid_output
        self.img_height = img_height
        self.img_width = img_width

        self.img_pe_col = nn.Parameter(torch.randn(img_height, 1, embed_dim))
        self.img_pe_row = nn.Parameter(torch.randn(img_width, 1, embed_dim))
        self.txt_pe = nn.Parameter(torch.randn(txt_max_length, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.text_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=txt_vocab_size, embedding_dim=embed_dim)
        )

        self.img_lin_proj = nn.Linear(embed_dim, embed_dim)

        self.tr_encoder_blocks = nn.ModuleList([
            TrEncoderBlock(n_features=embed_dim,
                           n_attn_heads=n_attn_heads,
                           n_hidden=hidden_dim,
                           dropout_prob=dropout_prob)
            for _ in range(num_blocks)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self.to(self.device)

    def forward(self, img_tokens, txt_tokens):
        seq_len, batch, emb = img_tokens.size()

        x = self.img_lin_proj(img_tokens)
        pe_column = self.img_pe_col.repeat(self.img_width, 1, 1)
        pe_row = self.img_pe_row.repeat_interleave(self.img_height, dim=0)
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


if __name__=="__main__":

    model = TrMatcher(
        img_height=CONFIG.hidden_height,
        img_width=CONFIG.hidden_width,
        embed_dim=CONFIG.vocab_size,
        txt_max_length=CONFIG.cond_seq_size,
        txt_vocab_size=CONFIG.cond_vocab_size,
        num_blocks=8,
        hidden_dim=512,
        n_attn_heads=8,
        dropout_prob=0.1,
        out_dim=1,
        sigmoid_output=True)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG.LR)

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

            with torch.no_grad():
                latent = dvae.ng_q_encode(img)

            b, emb, h, w = latent.size()
            x = latent.view(b, emb, -1).permute(2, 0, 1)

            pred_labels = model(x, txt)

            loss = F.binary_cross_entropy(pred_labels, match_labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()



