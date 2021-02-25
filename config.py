import os
import torch
import json


class Config:
    def __init__(self, local: bool = True):
        if local:
            root_dir = "/home/andrey/Aalto/TA-VQVAE/"
            self.BATCH_SIZE = 8
        else:
            root_dir = "/u/82/sukhoba1/unix/Desktop/TA-VQVAE"
            self.BATCH_SIZE = 64
        self.root_img_path = os.path.join(os.path.normpath(root_dir), "data/CUB/CUB_200_2011/images")
        self.root_text_path = os.path.join(os.path.normpath(root_dir), "data/CUB/text")
        self.imgs_list_file_path = os.path.join(os.path.normpath(root_dir), "data/CUB/CUB_200_2011/images.txt")
        self.save_model_path = os.path.join(os.path.normpath(root_dir), "models/vqvae_e256x4069/")
        self.vqvae_num_embeddings = 4069
        self.vqvae_embedding_dim = 256
        self.vqvae_commitment_cost = 0.25
        self.vqvae_decay = 0.99
        self.vqvae_num_x2downsamples = 3
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_EPOCHS = 500
        self.LR = 1e-3

    def save_config(self):
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        info = {
            "root_img_path": self.root_img_path,
            "root_text_path": self.root_text_path,
            "imgs_list_file_path": self.imgs_list_file_path,
            "save_model_path": self.save_model_path,
            "vqvae_num_embeddings": self.vqvae_num_embeddings,
            "vqvae_embedding_dim": self.vqvae_embedding_dim,
            "vqvae_commitment_cost": self.vqvae_commitment_cost,
            "vqvae_decay": self.vqvae_decay,
            "vqvae_num_x2downsamples": self.vqvae_num_x2downsamples,
            "BATCH_SIZE": self.BATCH_SIZE,
            "NUM_EPOCHS": self.NUM_EPOCHS,
            "LR": self.LR
        }
        save_path = os.path.join(self.save_model_path, 'config.json')
        with open(save_path, 'w') as outfile:
            json.dump(info, outfile, indent=4)

if __name__ == '__main__':
    c = Config(True)
    c.save_config()
