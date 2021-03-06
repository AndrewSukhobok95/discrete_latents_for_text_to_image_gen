import os
import torch
import json


class Config:
    def __init__(self,
                 local: bool = True,
                 model_path: str = "models/vqvae/"):
        if local:
            root_dir = "/home/andrey/Aalto/TA-VQVAE/"
            self.BATCH_SIZE = 8
        else:
            root_dir = "/u/82/sukhoba1/unix/Desktop/TA-VQVAE/"
            self.BATCH_SIZE = 128
        self.save_model_path = os.path.join(os.path.normpath(root_dir), model_path)
        self.root_img_path = os.path.join(os.path.normpath(root_dir), "data/CUB/CUB_200_2011/images")
        self.root_text_path = os.path.join(os.path.normpath(root_dir), "data/CUB/text")
        self.imgs_list_file_path = os.path.join(os.path.normpath(root_dir), "data/CUB/CUB_200_2011/images.txt")
        self.img_size = 128
        self.vqvae_num_embeddings = 4096
        self.vqvae_embedding_dim = 256
        self.vqvae_commitment_cost = 0.25
        self.vqvae_decay = 0.99
        self.vqvae_num_x2downsamples = 2
        self.vqvae_num_residual_layers = 4
        self.text_rebuild_num_residual_layers = 4
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_EPOCHS = 3000
        self.LR = 0.01
        self.LR_gamma = 0.1
        self.step_LR_milestones = [10, 100, 250]

    def save_config(self):
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        info = {
            "root_img_path": self.root_img_path,
            "root_text_path": self.root_text_path,
            "imgs_list_file_path": self.imgs_list_file_path,
            "save_model_path": self.save_model_path,
            "img_size": self.img_size,
            "vqvae_num_embeddings": self.vqvae_num_embeddings,
            "vqvae_embedding_dim": self.vqvae_embedding_dim,
            "vqvae_commitment_cost": self.vqvae_commitment_cost,
            "vqvae_decay": self.vqvae_decay,
            "vqvae_num_x2downsamples": self.vqvae_num_x2downsamples,
            "vqvae_num_residual_layers": self.vqvae_num_residual_layers,
            "text_rebuild_num_residual_layers": self.text_rebuild_num_residual_layers,
            "BATCH_SIZE": self.BATCH_SIZE,
            "NUM_EPOCHS": self.NUM_EPOCHS,
            "LR": self.LR,
            "LR_gamma": self.LR_gamma,
            "step_LR_milestones": self.step_LR_milestones
        }
        save_path = os.path.join(self.save_model_path, 'config.json')
        with open(save_path, 'w') as outfile:
            json.dump(info, outfile, indent=4)

    def load_config(self):
        save_path = os.path.join(self.save_model_path, 'config.json')
        with open(save_path, 'r') as file:
            info = json.load(file)
        self.img_size = info["img_size"]
        self.vqvae_num_embeddings = info["vqvae_num_embeddings"]
        self.vqvae_embedding_dim = info["vqvae_embedding_dim"]
        self.vqvae_commitment_cost = info["vqvae_commitment_cost"]
        self.vqvae_decay = info["vqvae_decay"]
        self.vqvae_num_x2downsamples = info["vqvae_num_x2downsamples"]
        self.vqvae_num_residual_layers = info["vqvae_num_residual_layers"]
        self.text_rebuild_num_residual_layers = info["text_rebuild_num_residual_layers"]


if __name__ == '__main__':
    c = Config(local=True, model_path="models/vqvae/")
    c.save_config()
    c.load_config()
