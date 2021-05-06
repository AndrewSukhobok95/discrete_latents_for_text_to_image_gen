import os
import torch
import torch.nn as nn

from modules.vqvae.model import VQVAE
from modules.tavqvae.generator.text_rebuild_block import TextRebuildBlock


class Generator(nn.Module):
    def __init__(self,
                 num_embeddings,
                 img_embedding_dim,
                 text_embedding_dim,
                 commitment_cost,
                 decay=0.0,
                 num_x2downsamples=2,
                 num_resid_downsample_layers=1,
                 num_resid_bottleneck_layers=2,
                 text_rebuild_num_residual_layers=2,
                 use_batch_norm=False,
                 vqvae_use_conv1x1=False):
        super(Generator, self).__init__()
        self.vqvae = VQVAE(num_embeddings=num_embeddings,
                           embedding_dim=img_embedding_dim,
                           commitment_cost=commitment_cost,
                           decay=decay,
                           num_x2downsamples=num_x2downsamples,
                           num_resid_downsample_layers=num_resid_downsample_layers,
                           num_resid_bottleneck_layers=num_resid_bottleneck_layers,
                           use_batch_norm=use_batch_norm,
                           use_conv1x1=vqvae_use_conv1x1)
        self.rebuild_block = TextRebuildBlock(img_embedding_dim,
                                              text_embedding_dim,
                                              text_rebuild_num_residual_layers,
                                              bias=False)

    def forward(self, imgh, texth, text_mask=None):
        s = self.vqvae.encode(imgh)
        s_rebuild, mask = self.rebuild_block(s, texth, text_mask)
        vq_loss, quantized, perplexity, encoding_info = self.vqvae.quantize(s_rebuild)
        x_recon = self.vqvae.decode(quantized)
        return x_recon, mask, quantized, encoding_info, perplexity

    def get_rebuild_parameters(self):
        return self.rebuild_block.parameters()

    def load_vqvae_weights(self, root_path, model_name):
        self.vqvae.load_model(root_path, model_name)

    def save_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        encoder_path = os.path.join(root_path, model_name + "_encoder.pth")
        decoder_path = os.path.join(root_path, model_name + "_decoder.pth")
        quantizer_path = os.path.join(root_path, model_name + "_quantizer.pth")
        text_rebuild_path = os.path.join(root_path, model_name + "_text_rebuild.pth")
        torch.save(self.vqvae.encoder.state_dict(), encoder_path)
        torch.save(self.vqvae.decoder.state_dict(), decoder_path)
        torch.save(self.vqvae.quantizer.state_dict(), quantizer_path)
        torch.save(self.rebuild_block.state_dict(), text_rebuild_path)

    def load_model(self, root_path, model_name, map_location=torch.device('cpu')):
        encoder_path = os.path.join(root_path, model_name + "_encoder.pth")
        decoder_path = os.path.join(root_path, model_name + "_decoder.pth")
        quantizer_path = os.path.join(root_path, model_name + "_quantizer.pth")
        rebuild_block_path = os.path.join(root_path, model_name + "_text_rebuild.pth")
        self.vqvae.encoder.load_state_dict(torch.load(encoder_path, map_location=map_location))
        self.vqvae.decoder.load_state_dict(torch.load(decoder_path, map_location=map_location))
        self.vqvae.quantizer.load_state_dict(torch.load(quantizer_path, map_location=map_location))
        self.rebuild_block.load_state_dict(torch.load(rebuild_block_path, map_location=map_location))


