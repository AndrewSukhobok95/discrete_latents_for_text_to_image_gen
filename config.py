import torch


class Config:
    def __init__(self, local: bool = True):
        if local:
            self.root_img_path = "/home/andrey/Aalto/TA-VQVAE/data/CUB/CUB_200_2011/images"
            self.root_text_path = "/home/andrey/Aalto/TA-VQVAE/data/CUB/text"
            self.imgs_list_file_path = "/home/andrey/Aalto/TA-VQVAE/data/CUB/CUB_200_2011/images.txt"
            self.save_model_path = "/home/andrey/Aalto/TA-VQVAE/models/"
            self.BATCH_SIZE = 8
        else:
            self.root_img_path = "/u/82/sukhoba1/unix/Desktop/TA-VQVAE/data/CUB/CUB_200_2011/images"
            self.root_text_path = "/u/82/sukhoba1/unix/Desktop/TA-VQVAE/data/CUB/text"
            self.imgs_list_file_path = "/u/82/sukhoba1/unix/Desktop/TA-VQVAE/data/CUB/CUB_200_2011/images.txt"
            self.save_model_path = "/u/82/sukhoba1/unix/Desktop/TA-VQVAE/models/"
            self.BATCH_SIZE = 64
        self.vqvae_num_embeddings = 1024
        self.vqvae_embedding_dim = 256
        self.vqvae_commitment_cost = 0.25
        self.vqvae_decay = 0.99
        self.vqvae_num_x2downsamples = 3
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_EPOCHS = 1000
        self.LR = 1e-3
