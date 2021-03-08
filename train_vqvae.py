import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from datasets.cub import CubDataset, cub_collate
from modules.vqvae.model import VQVAE
from config import Config


CONFIG = Config(local=True, model_path="models/vqvae_i128_e256x8192_v2/")
CONFIG.save_config()
CONFIG.print_config_info()

writer = SummaryWriter()

cub_dataset = CubDataset(root_img_path=CONFIG.root_img_path,
                         root_text_path=CONFIG.root_text_path,
                         imgs_list_file_path=CONFIG.imgs_list_file_path,
                         img_size=CONFIG.img_size)
train_length = int(len(cub_dataset) * 0.99)
test_length = len(cub_dataset) - train_length
train_dataset, test_dataset = torch.utils.data.random_split(
    cub_dataset, lengths=[train_length, test_length], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=CONFIG.BATCH_SIZE,
                          shuffle=True,
                          collate_fn=cub_collate)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=CONFIG.BATCH_SIZE,
                         shuffle=True,
                         collate_fn=cub_collate)

model = VQVAE(num_embeddings=CONFIG.vqvae_num_embeddings,
              embedding_dim=CONFIG.vqvae_embedding_dim,
              commitment_cost=CONFIG.vqvae_commitment_cost,
              decay=CONFIG.vqvae_decay,
              num_x2downsamples=CONFIG.vqvae_num_x2downsamples,
              num_residual_layers=CONFIG.vqvae_num_residual_layers,
              use_batch_norm=True,
              use_conv1x1=True)

optimizer = optim.Adam(model.parameters(), lr=CONFIG.LR)
lr_scheduler = MultiStepLR(optimizer, milestones=CONFIG.step_LR_milestones, gamma=CONFIG.LR_gamma)


def validate():
    n = 0
    test_loss = torch.tensor(0)
    for imgs, _ in test_loader:
        imgs = imgs.to(CONFIG.DEVICE)
        with torch.no_grad():
            vq_loss, data_recon, perplexity = model(imgs)
            recon_error = F.mse_loss(data_recon, imgs)
            loss = recon_error + vq_loss
        test_loss += loss
        n += 1
    return test_loss / n


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

            test_loss = validate()

            _last_lr = lr_scheduler.get_last_lr()
            print("Epoch: {} Iter: {} Loss: {} Test Loss: {} LR: {}".format(
                epoch, iteration, loss.item(), test_loss.item(), _last_lr))

            tag_scalar_dict = {
                "train": loss.item(),
                "test": test_loss.item()
            }
            writer.add_scalar('Loss', tag_scalar_dict, iteration)
            writer.add_scalar('VQLoss', vq_loss.item(), iteration)
            writer.add_scalar('ReconLoss', recon_error.item(), iteration)
            writer.add_scalar('Perplexity', perplexity.item(), iteration)

            iteration += 1

        img_grid = torchvision.utils.make_grid(imgs[:8, :, :, :].detach().cpu())
        writer.add_image('BirdImg', img_grid, epoch)

        img_recon_grid = torchvision.utils.make_grid(data_recon[:8, :, :, :].detach().cpu())
        writer.add_image('BirdImgReconstruction', img_recon_grid, epoch)

        lr_scheduler.step()

        model.save_model(root_path=CONFIG.save_model_path, model_name="VQVAE")

    writer.close()
