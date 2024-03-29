import os
import torch
import torch.nn as nn

from modules.dalle_dvae.model import DVAE
from modules.tadvae.generator.text_rebuild_block import TextRebuildBlock


class Generator(nn.Module):
    def __init__(self,
                 img_embedding_dim,
                 text_embedding_dim,
                 n_trd_blocks=4,
                 num_trd_block_for_mask=3,
                 n_attn_heads=4,
                 linear_hidden_dim=1024,
                 dropout_prob=0.1,
                 n_img_hidden_positions=1024):
        super(Generator, self).__init__()
        self.dvae = DVAE()
        self.text_rebuild_block = TextRebuildBlock(img_hidden_dim=img_embedding_dim,
                                                   txt_hidden_dim=text_embedding_dim,
                                                   n_trd_blocks=n_trd_blocks,
                                                   num_trd_block_for_mask=num_trd_block_for_mask,
                                                   n_attn_heads=n_attn_heads,
                                                   linear_hidden_dim=linear_hidden_dim,
                                                   dropout_prob=dropout_prob,
                                                   n_img_hidden_positions=n_img_hidden_positions)

    def forward(self, img, txt_h, txt_pad_mask, return_mask=False):
        # z = self.dvae.encode(img)
        # z_new, z_mask = self.text_rebuild_block(z, txt_h, txt_pad_mask)
        # z_onehot = self.dvae.quantize(z_new)
        # img_recon = self.dvae.decode(z_onehot)
        z_new, z_onehot, z_mask = self.encode(img, txt_h, txt_pad_mask)
        img_recon = self.decode(z_onehot)
        if return_mask:
            return img_recon, z_mask
        return img_recon

    def encode(self, img, txt_h, txt_pad_mask):
        z = self.dvae.encode(img)
        z_new, z_mask = self.text_rebuild_block(z, txt_h, txt_pad_mask)
        z_onehot = self.dvae.quantize(z_new)
        return z_new, z_onehot, z_mask

    def decode(self, z):
        img_recon = self.dvae.decode(z)
        return img_recon

    def load_dvae_weights(self, root_path, model_name):
        self.dvae.load_model(root_path, model_name)

    def get_rebuild_params(self):
        return self.text_rebuild_block.parameters()

    def save_rebuild_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        path = os.path.join(root_path, model_name + "_text_rebuild.pth")
        torch.save(self.text_rebuild_block.state_dict(), path)

    def load_rebuild_model(self, root_path, model_name):
        path = os.path.join(root_path, model_name + "_text_rebuild.pth")
        self.text_rebuild_block.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


