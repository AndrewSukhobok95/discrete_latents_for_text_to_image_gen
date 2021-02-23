import torch


class Config:
    root_img_path = "/home/andrey/Aalto/TA-VQVAE/data/CUB/CUB_200_2011/images"
    root_text_path = "/home/andrey/Aalto/TA-VQVAE/data/CUB/text"
    imgs_list_file_path = "/home/andrey/Aalto/TA-VQVAE/data/CUB/CUB_200_2011/images.txt"
    save_model_path = "/home/andrey/Aalto/TA-VQVAE/models/"
    vqvae_num_embeddings = 1024
    vqvae_embedding_dim = 256
    vqvae_commitment_cost = 0.25
    vqvae_decay = 0.99
    vqvae_num_x2downsamples = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    NUM_EPOCHS = 1000
    LR = 1e-3
