import os
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config_reader import ConfigReader
from datasets.triple_mnist import TripleMnistDataset
from modules.dvae.model import DVAE
from train_utils.dvae_utils import KLD_uniform_loss, TemperatureAnnealer, KLDWeightAnnealer


# CONFIG = ConfigReader(config_path="/home/andrey/Aalto/thesis/TA-VQVAE/configs/dvae_triplemnist_local.yaml")
CONFIG = ConfigReader(config_path="/u/82/sukhoba1/unix/Desktop/TA-VQVAE/configs/dvae_triplemnist_remote.yaml")
CONFIG.print_config_info()

writer = SummaryWriter()


dataset = TripleMnistDataset(
    root_img_path=CONFIG.root_img_path)

train_loader = DataLoader(
    dataset=dataset,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=True)

model = DVAE(in_channels=CONFIG.in_channels,
             vocab_size=CONFIG.vocab_size,
             num_x2downsamples=CONFIG.num_x2downsamples,
             num_resids_downsample=CONFIG.num_resids_downsample,
             num_resids_bottleneck=CONFIG.num_resids_bottleneck)

optimizer = optim.Adam(model.parameters(), lr=CONFIG.LR)
lr_scheduler = MultiStepLR(optimizer, milestones=CONFIG.step_LR_milestones, gamma=CONFIG.LR_gamma)


if __name__ == '__main__':
    print("Device in use: {}".format(CONFIG.DEVICE))

    temp_annealer = TemperatureAnnealer(
        start_temp=CONFIG.temp_start,
        end_temp=CONFIG.temp_end,
        n_steps=CONFIG.temp_steps)

    kl_annealer = KLDWeightAnnealer(
        start_lambda=CONFIG.KLD_lambda_start,
        end_lambda=CONFIG.KLD_lambda_end,
        n_steps=CONFIG.KLD_lambda_steps)

    model.train()
    model.to(CONFIG.DEVICE)

    iteration = 0
    for epoch in range(CONFIG.NUM_EPOCHS):
        for x, _ in train_loader:
            x = x.to(CONFIG.DEVICE)

            optimizer.zero_grad()

            temp = temp_annealer.step(iteration)
            x_recon, z_dist = model(x, temp)

            recon_loss = F.binary_cross_entropy(x_recon, x)
            kld_loss = KLD_uniform_loss(z_dist)
            kl_weight = kl_annealer.step(iteration)

            loss = recon_loss - kl_weight * kld_loss

            loss.backward()
            optimizer.step()

            print("Epoch: {} Iter: {} Loss: {} KL Loss (weighted): {} Recon Loss {}".format(
                epoch, iteration, loss.item(), kl_weight * kld_loss.item(), recon_loss.item()))
            print("N codes used: {}".format(len(z_dist.detach().cpu().argmax(dim=1).view(-1).unique())))

            writer.add_scalar('loss/recon_loss', recon_loss.item(), iteration)
            writer.add_scalar('loss/kld_loss', kld_loss.item(), iteration)
            writer.add_scalar('rates/temperature', temp, iteration)
            writer.add_scalar('rates/kl_weight', kl_weight, iteration)

            iteration += 1

        img_grid = torchvision.utils.make_grid(x[:8, :, :, :].detach().cpu())
        writer.add_image('images/original', img_grid, epoch)
        img_recon_grid = torchvision.utils.make_grid(x_recon[:8, :, :, :].detach().cpu())
        writer.add_image('images/reconstruction', img_recon_grid, epoch)

        lr_scheduler.step()

        model.save_model(CONFIG.save_model_path, CONFIG.save_model_name)




