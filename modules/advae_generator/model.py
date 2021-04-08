import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.dvae.model import DVAE


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dvae = DVAE()

    def forward(self, img, txt_h, txt_pad_mask, return_mask=False):
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

    def load_dvae_model(self, root_path, model_name):
        self.dvae.load_model(root_path, model_name)

    # def save_rebuild_model(self, root_path, model_name):
    #     if not os.path.exists(root_path):
    #         os.makedirs(root_path)
    #     path = os.path.join(root_path, model_name + "_text_rebuild.pth")
    #     torch.save(self.text_rebuild_block.state_dict(), path)
    #
    # def load_rebuild_model(self, root_path, model_name):
    #     path = os.path.join(root_path, model_name + "_text_rebuild.pth")
    #     self.text_rebuild_block.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


