import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.cub import CubDataset, cub_collate
from modules.vqvae.model import VQVAE
from config import Config


CONFIG = Config(local=False, model_path="models/vqvae_i128_e512x8138/")
CONFIG.save_config()

writer = SummaryWriter()

train_dataset = CubDataset(root_img_path=CONFIG.root_img_path,
                           root_text_path=CONFIG.root_text_path,
                           imgs_list_file_path=CONFIG.imgs_list_file_path,
                           img_size=CONFIG.img_size)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=CONFIG.BATCH_SIZE,
                          shuffle=True,
                          collate_fn=cub_collate)

model = VQVAE(num_embeddings=CONFIG.vqvae_num_embeddings,
              embedding_dim=CONFIG.vqvae_embedding_dim,
              commitment_cost=CONFIG.vqvae_commitment_cost,
              decay=CONFIG.vqvae_decay,
              num_x2downsamples=CONFIG.vqvae_num_x2downsamples)
optimizer = optim.Adam(model.parameters(), lr=CONFIG.LR)


if __name__ == '__main__':
    print("Device in use: {}".format(CONFIG.DEVICE))

    model.train()
    model.to(CONFIG.DEVICE)

    iteration = 0
    for epoch in range(CONFIG.NUM_EPOCHS):
        for imgs, _ in train_loader:
            imgs = imgs.to(CONFIG.DEVICE)

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

        model.save_model(root_path=CONFIG.save_model_path, model_name="VQVAE")

    writer.close()
