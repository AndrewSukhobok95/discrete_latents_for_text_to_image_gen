import os
import torch
import torch.nn as nn
from modules.vqvae.blocks import Encoder, Decoder
from modules.vqvae.quantizer import VectorQuantizer, VectorQuantizerEMA


class VQVAE(nn.Module):
    def __init__(self,
                 img_channels,
                 num_embeddings,
                 embedding_dim,
                 commitment_cost,
                 decay=0.0,
                 num_x2downsamples=2,
                 num_resid_downsample_layers=1,
                 num_resid_bottleneck_layers=2,
                 bias=False,
                 use_batch_norm=False,
                 use_conv1x1=False):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels=img_channels,
                               out_channels=embedding_dim,
                               num_downsamples=num_x2downsamples,
                               num_resid_downsample=num_resid_downsample_layers,
                               num_resid_bottleneck=num_resid_bottleneck_layers,
                               bias=bias,
                               use_bn=use_batch_norm,
                               use_conv1x1=use_conv1x1)
        if decay > 0.0:
            self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(in_channels=embedding_dim,
                               out_channels=img_channels,
                               num_upsamples=num_x2downsamples,
                               num_resid_upsample=num_resid_downsample_layers,
                               num_resid_bottleneck=num_resid_bottleneck_layers,
                               bias=bias,
                               use_bn=use_batch_norm,
                               use_conv1x1=use_conv1x1)

    def forward(self, x):
        latent = self.encoder(x)
        vq_loss, quantized, perplexity, encoding_info = self.quantizer(latent)
        x_recon = self.decoder(quantized)
        return vq_loss, quantized, x_recon, perplexity

    def encode(self, x):
        z = self.encoder(x)
        return z

    def quantize(self, z):
        vq_loss, quantized, perplexity, encoding_info = self.quantizer(z)
        return vq_loss, quantized, perplexity, encoding_info

    def decode(self, z, sigmoid_activation=True):
        x_recon = self.decoder(z)
        if sigmoid_activation:
            return torch.sigmoid(x_recon)
        return x_recon

    def get_encoder_decoder_params(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_quantizer_params(self):
        return self.quantizer.parameters()

    def encode_and_quantize(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, encoding_info = self.quantizer(z)
        return quantized, encoding_info

    def decode_by_index_mask(self, index_mask):
        z = self.quantizer.get_by_index(index_mask).permute(0, 3, 1, 2)
        x_recon = self.decoder(z)
        return x_recon

    def load_model(self, root_path, model_name, map_location=torch.device('cpu')):
        encoder_path = os.path.join(root_path, model_name + "_encoder.pth")
        decoder_path = os.path.join(root_path, model_name + "_decoder.pth")
        quantizer_path = os.path.join(root_path, model_name + "_quantizer.pth")
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=map_location))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=map_location))
        self.quantizer.load_state_dict(torch.load(quantizer_path, map_location=map_location))

    def save_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        encoder_path = os.path.join(root_path, model_name + "_encoder.pth")
        decoder_path = os.path.join(root_path, model_name + "_decoder.pth")
        quantizer_path = os.path.join(root_path, model_name + "_quantizer.pth")
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
        torch.save(self.quantizer.state_dict(), quantizer_path)


