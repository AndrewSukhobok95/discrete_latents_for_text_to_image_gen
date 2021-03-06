import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel
from datasets.cub import CubDataset, CubCollater
from config import Config
from modules.tavqvae.generator import Generator
from modules.tavqvae.tagan_discriminator import Discriminator


CONFIG = Config(local=True, model_path="models/tavqvae_e512x8138/")
CONFIG.save_config()

writer = SummaryWriter()

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

train_dataset = CubDataset(root_img_path=CONFIG.root_img_path,
                           root_text_path=CONFIG.root_text_path,
                           imgs_list_file_path=CONFIG.imgs_list_file_path,
                           img_size=CONFIG.img_size)
collater = CubCollater(tokenizer=bert_tokenizer)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=CONFIG.BATCH_SIZE,
                          shuffle=True,
                          collate_fn=collater.collate)


G = Generator(num_embeddings=CONFIG.vqvae_num_embeddings,
              img_embedding_dim=CONFIG.vqvae_embedding_dim,
              text_embedding_dim=bert_model.config.hidden_size,
              commitment_cost=CONFIG.vqvae_commitment_cost,
              decay=CONFIG.vqvae_decay,
              num_x2downsamples=CONFIG.vqvae_num_x2downsamples,
              vqvae_num_residual_layers=CONFIG.vqvae_num_residual_layers,
              text_rebuild_num_residual_layers=CONFIG.text_rebuild_num_residual_layers)
D = Discriminator(text_embedding_dim=bert_model.config.hidden_size)

optimizer_G = optim.Adam(G.get_rebuild_parameters(), lr=CONFIG.LR)
optimizer_D = optim.Adam(D.parameters(), lr=CONFIG.LR)


if __name__ == '__main__':
    print("Device in use: {}".format(CONFIG.DEVICE))

    G.train()
    G.to(CONFIG.DEVICE)
    D.train()
    D.to(CONFIG.DEVICE)

    iteration = 0
    for epoch in range(CONFIG.NUM_EPOCHS):
        for imgs, text_info in train_loader:
            token_tensor, token_type_tensor, mask_tensor = text_info

            print()




