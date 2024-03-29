import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.dvae.blocks import Encoder, Decoder
from modules.dvae.funcs import ng_quantize


class DVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 vocab_size,
                 num_x2downsamples=2,
                 num_resids_downsample=2,
                 num_resids_bottleneck=2,
                 hidden_dim=None,
                 device=torch.device('cpu')):
        super(DVAE, self).__init__()
        self.device = device
        self.encoder = Encoder(in_channels=in_channels,
                               out_channels=vocab_size,
                               num_x2downsamples=num_x2downsamples,
                               num_resids_downsample=num_resids_downsample,
                               num_resids_bottleneck=num_resids_bottleneck,
                               hidden_channels=hidden_dim,
                               bias=True,
                               use_bn=True)
        self.decoder = Decoder(in_channels=vocab_size,
                               out_channels=in_channels,
                               num_x2upsamples=num_x2downsamples,
                               num_resids_upsample=num_resids_downsample,
                               num_resids_bottleneck=num_resids_bottleneck,
                               hidden_channels=hidden_dim,
                               bias=True,
                               use_bn=True)
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def encode(self, x):
        z_logits = self.encoder(x)
        return z_logits

    def gumbel_quantize(self, z_logits, tau=1 / 16, hard=False):
        return F.gumbel_softmax(z_logits, tau=tau, hard=hard, dim=1)

    def decode(self, z):
        x_rec = self.decoder(z)
        return x_rec

    def sm_encode(self, x):
        z_logits = self.encoder(x)
        z = F.softmax(z_logits, dim=1)
        return z

    def ng_q_encode(self, x):
        with torch.no_grad():
            z_logits = self.encoder(x)
            z = ng_quantize(z_logits)
        return z

    def ng_q_decode(self, z_logits):
        with torch.no_grad():
            z = ng_quantize(z_logits)
            x_rec = self.decoder(z)
        return x_rec

    def q_encode(self, x, tau=1/16, hard=False):
        z_logits = self.encoder(x)
        z = F.gumbel_softmax(z_logits, tau=tau, hard=hard, dim=1)
        return z

    def q_decode(self, z_logits, tau=1/16, hard=False):
        z = F.gumbel_softmax(z_logits, tau=tau, hard=hard, dim=1)
        x_rec = self.decoder(z)
        return x_rec

    def forward(self, x, tau=1/16, hard=False):
        z_logits = self.encode(x)
        z = self.gumbel_quantize(z_logits, tau=tau, hard=hard)
        x_rec = self.decode(z)
        return x_rec, z_logits, z

    def get_reconstruction(self, x):
        z_logits = self.encode(x)
        z = ng_quantize(z_logits)
        x_rec = self.decoder(z)
        return x_rec

    def load_model(self, root_path, model_name, map_location=torch.device('cpu')):
        encoder_path = os.path.join(root_path, model_name + "_encoder.pth")
        decoder_path = os.path.join(root_path, model_name + "_decoder.pth")
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=map_location))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=map_location))

    def save_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        encoder_path = os.path.join(root_path, model_name + "_encoder.pth")
        decoder_path = os.path.join(root_path, model_name + "_decoder.pth")
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def show_model_architecture(self):
        print("DVAE architecture:")
        print()
        self.encoder.show_model_architecture()
        print()
        self.decoder.show_model_architecture()
        print()
