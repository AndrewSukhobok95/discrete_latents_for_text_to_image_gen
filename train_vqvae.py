import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataload.cub import CubDataset, cub_collate
from vqvae.model import VQVAE
from config import Config


writer = SummaryWriter()

train_dataset = CubDataset(root_img_path=Config.root_img_path,
                           root_text_path=Config.root_text_path,
                           imgs_list_file_path=Config.imgs_list_file_path)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=Config.BATCH_SIZE,
                          shuffle=True,
                          collate_fn=cub_collate)

model = VQVAE(num_embeddings=Config.vqvae_num_embeddings,
              embedding_dim=Config.vqvae_embedding_dim,
              commitment_cost=Config.vqvae_commitment_cost,
              decay=Config.vqvae_decay,
              num_x2downsamples=Config.vqvae_num_x2downsamples)
optimizer = optim.Adam(model.parameters(), lr=Config.LR)


if __name__ == '__main__':
    model.train()
    model.to(Config.DEVICE)

    iteration = 0
    for epoch in range(Config.NUM_EPOCHS):
        for imgs, _ in train_loader:
            imgs = imgs.to(Config.DEVICE)

            vq_loss, data_recon, perplexity = model(imgs)
            recon_error = F.mse_loss(data_recon, imgs)
            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            print("Epoch: {} Iter: {} Loss: {}".format(epoch, iteration, loss.item()))

            writer.add_scalar('Loss', loss.item(), iteration)
            writer.add_scalar('VQLoss', vq_loss.item(), iteration)
            writer.add_scalar('ReconLoss', recon_error.item(), iteration)
            writer.add_scalar('Perplexity', perplexity.item(), iteration)
            iteration += 1

        model.save_model(root_path=Config.save_model_path, model_name="VQVAE")

    writer.close()
