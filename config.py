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
            root_dir = "/u/82/sukhoba1/unix/Desktop/projects/TA-VQVAE/"
            self.BATCH_SIZE = 64
        self.save_model_path = os.path.join(os.path.normpath(root_dir), model_path)
        self.root_img_path = os.path.join(os.path.normpath(root_dir), "data/CUB/CUB_200_2011/images")
        self.root_text_path = os.path.join(os.path.normpath(root_dir), "data/CUB/text")
        self.imgs_list_file_path = os.path.join(os.path.normpath(root_dir), "data/CUB/CUB_200_2011/images.txt")
        self.img_size = 128
        self.vqvae_num_embeddings = 8192  # 8192 4096
        self.vqvae_embedding_dim = 256
        self.vqvae_commitment_cost = 0.25
        self.vqvae_decay = 0.99
        self.vqvae_num_x2downsamples = 2
        self.vqvae_num_downsample_residual_layers = 2
        self.vqvae_num_residual_layers = 4
        self.text_rebuild_num_residual_layers = 6
        self.tagan_lambda_cond_loss = 10
        self.tagan_lambda_recon_loss = 0.2
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_EPOCHS = 3000
        self.LR = 0.01
        self.quantizer_LR = 0.1
        self.LR_gamma = 0.1
        self.step_LR_milestones = [50, 150, 200]

    def save_config(self):
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        config_params = vars(self)
        for k, v in config_params.items():
            if isinstance(v, torch.device):
                config_params[k] = str(v)
        save_path = os.path.join(self.save_model_path, 'config.json')
        with open(save_path, 'w') as outfile:
            json.dump(config_params, outfile, indent=4)

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

    def print_config_info(self):
        row_format ="{:<40}" * 2
        config_params = vars(self)
        keys = sorted(config_params.keys())
        for k in keys:
            print(row_format.format(k, str(config_params[k])))


if __name__ == '__main__':
    c = Config(local=True, model_path="models/vqvae/")
    c.print_config_info()
    c.save_config()
    c.load_config()
