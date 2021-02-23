import os
import torch
import torch.nn as nn
from vqvae.modules import Encoder, Decoder
from vqvae.quantizer import VectorQuantizer, VectorQuantizerEMA


class VQVAE(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 commitment_cost,
                 decay=0.0,
                 num_x2downsamples=2):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels=3, out_channels=embedding_dim, num_downsamples=num_x2downsamples)
        if decay > 0.0:
            self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(in_channels=embedding_dim, out_channels=3, num_upsamples=num_x2downsamples)

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity

    def save_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        encoder_path = os.path.join(root_path, model_name + "_encoder.pth")
        torch.save(self.encoder.state_dict(), encoder_path)
        decoder_path = os.path.join(root_path, model_name + "_decoder.pth")
        torch.save(self.decoder.state_dict(), decoder_path)
        quantizer_path = os.path.join(root_path, model_name + "_quantizer.pth")
        torch.save(self.quantizer.state_dict(), quantizer_path)


