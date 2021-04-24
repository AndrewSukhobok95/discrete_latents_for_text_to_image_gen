import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.dvae.model import Decoder
from modules.generative_lsdvae.blocks import LatentGenerator


class Generator(nn.Module):
    def __init__(self,
                 noise_dim,
                 hidden_height,
                 hidden_width,
                 hidden_vocab_size,
                 out_channels,
                 num_tr_blocks,
                 dvae_num_x2upsamples=2,
                 dvae_num_resids_upsample=3,
                 dvae_num_resids_bottleneck=4,
                 dvae_hidden_dim=64,
                 tr_hidden_dim=64,
                 tr_n_attn_heads=8,
                 dropout_prob=0.1):
        super(Generator, self).__init__()

        self.lat_gen = LatentGenerator(
            noise_dim=noise_dim,
            hidden_height=hidden_height,
            hidden_width=hidden_width,
            hidden_vocab_size=hidden_vocab_size,
            num_tr_blocks=num_tr_blocks,
            tr_hidden_dim=tr_hidden_dim,
            tr_n_attn_heads=tr_n_attn_heads,
            dropout_prob=dropout_prob)

        self.dvae_decoder = Decoder(
            in_channels=hidden_vocab_size,
            out_channels=out_channels,
            num_x2upsamples=dvae_num_x2upsamples,
            num_resids_upsample=dvae_num_resids_upsample,
            num_resids_bottleneck=dvae_num_resids_bottleneck,
            hidden_channels=dvae_hidden_dim,
            bias=True,
            use_bn=True)
        self.dvae_freeze_mode()

    def forward(self, noise, condition, tau=1/16, hard=False):
        z_logits = self.lat_gen(noise, condition)
        z = self.quantize(z_logits, tau=tau, hard=hard)
        x = self.dvae_decoder(z)
        return x, z_logits, z

    def quantize(self, z_logits, tau=1/16, hard=False):
        return F.gumbel_softmax(z_logits, tau=tau, hard=hard, dim=1)

    def dvae_freeze_mode(self):
        self.dvae_decoder.eval()
        for param in self.dvae_decoder.parameters():
            param.requires_grad = False

    def parameters_lat_generator(self):
        return self.lat_gen.parameters()

    def save_latent_generator(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        path = os.path.join(root_path, model_name + "_lat_gen.pth")
        torch.save(self.lat_gen.state_dict(), path)

    def load_latent_generator(self, root_path, model_name, map_location=torch.device('cpu')):
        path = os.path.join(root_path, model_name + "_lat_gen.pth")
        self.lat_gen.load_state_dict(torch.load(path, map_location=map_location))

    def load_dvae_decoder(self, root_path, model_name, map_location=torch.device('cpu')):
        path = os.path.join(root_path, model_name + "_decoder.pth")
        self.dvae_decoder.load_state_dict(torch.load(path, map_location=map_location))






