import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.dalle_dvae.encoder import Encoder
from modules.dalle_dvae.decoder import Decoder
from modules.dalle_dvae.utils import map_pixels, unmap_pixels


class DVAE(nn.Module):
    def __init__(self):
        super(DVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        x = map_pixels(x)
        z_logits = self.encoder(x)
        return z_logits

    def quantize(self, z_logits):
        z = torch.argmax(z_logits, dim=1)
        z = F.one_hot(z, num_classes=self.encoder.vocab_size).permute(0, 3, 1, 2).float()
        return z

    def decode(self, z):
        x_stats = self.decoder(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        return x_rec

    def forward(self, x):
        z_logits = self.encode(x)
        z = self.quantize(z_logits)
        x_rec = self.decode(z)
        return x_rec

    def load_model(self, root_path, model_name, map_location=torch.device('cpu')):
        encoder_path = os.path.join(root_path, model_name + "_encoder.pth")
        decoder_path = os.path.join(root_path, model_name + "_decoder.pth")
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=map_location))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=map_location))



