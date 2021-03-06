import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.vqvae.model import VQVAE
from modules.tavqvae.text_rebuild import TextRebuildBlock


class Generator(nn.Module):
    def __init__(self,
                 num_embeddings,
                 img_embedding_dim,
                 text_embedding_dim,
                 commitment_cost,
                 decay=0.0,
                 num_x2downsamples=2,
                 vqvae_num_residual_layers=2,
                 text_rebuild_num_residual_layers=2,
                 use_batch_norm=False,
                 vqvae_use_conv1x1=False):
        super(Generator, self).__init__()
        self.vqvae = VQVAE(num_embeddings,
                           img_embedding_dim,
                           commitment_cost,
                           decay,
                           num_x2downsamples,
                           vqvae_num_residual_layers,
                           use_batch_norm=use_batch_norm,
                           use_conv1x1=vqvae_use_conv1x1)
        self.rebuild_block = TextRebuildBlock(img_embedding_dim,
                                              text_embedding_dim,
                                              text_rebuild_num_residual_layers,
                                              bias=False)

    def forward(self, imgh, texth, text_mask=None):
        s = self.vqvae.encode(imgh)
        s_rebuild = self.rebuild_block(s, texth, text_mask)
        quantized, perplexity, _ = self.vqvae.quantize(s_rebuild)
        x_recon = self.vqvae.decode(quantized)
        return x_recon, perplexity

    def get_rebuild_parameters(self):
        return self.rebuild_block.parameters()


