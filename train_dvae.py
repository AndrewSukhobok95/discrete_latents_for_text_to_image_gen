import os
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import argparse
from torch.utils.tensorboard import SummaryWriter
from config_reader import ConfigReader
from datasets.mnist_loader import MNISTData
from datasets.cub_loader import CUBData
from modules.dvae.model import DVAE
from utilities.dvae_utils import TemperatureAnnealer, \
    SigmoidAnnealer, FixedAnnealer,\
    KLD_codes_uniform_loss, LinearAnnealer

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-cn', '--configname', action='store', type=str, required=True)
args = argument_parser.parse_args()

# config_dir = '/home/andrey/Aalto/thesis/TA-VQVAE/configs/'
config_dir = '/u/82/sukhoba1/unix/Desktop/TA-VQVAE/configs/'
config_name = args.configname
config_path = os.path.join(config_dir, config_name)

CONFIG = ConfigReader(config_path=config_path)
CONFIG.print_config_info()

writer = SummaryWriter(comment='_' + config_name)

if CONFIG.dataset == 'mnist':
    data_source = MNISTData(
        img_type=CONFIG.type,
        root_path=CONFIG.root_path,
        batch_size=CONFIG.BATCH_SIZE)
elif CONFIG.dataset == 'cub':
    data_source = CUBData(
        img_type=CONFIG.dataset_type,
        root_path=CONFIG.root_path,
        batch_size=CONFIG.BATCH_SIZE,
        prct_train_split=0.99)

train_loader = data_source.get_train_loader()

model = DVAE(in_channels=CONFIG.in_channels,
             vocab_size=CONFIG.vocab_size,
             num_x2downsamples=CONFIG.num_x2downsamples,
             num_resids_downsample=CONFIG.num_resids_downsample,
             num_resids_bottleneck=CONFIG.num_resids_bottleneck,
             hidden_dim=CONFIG.hidden_dim)

model.show_model_architecture()

optimizer = optim.Adam(model.parameters(), lr=CONFIG.LR)
lr_scheduler = MultiStepLR(optimizer, milestones=CONFIG.step_LR_milestones, gamma=CONFIG.LR_gamma)

temp_annealer = TemperatureAnnealer(
    start_temp=CONFIG.temp_start,
    end_temp=CONFIG.temp_end,
    n_steps=CONFIG.temp_steps)

if CONFIG.klU_type == 'sigmoid':
    klU_annealer = SigmoidAnnealer(
        start_lambda=CONFIG.klU_start,
        end_lambda=CONFIG.klU_end,
        n_steps=CONFIG.klU_steps)
elif CONFIG.klU_type == 'linear':
    klU_annealer = LinearAnnealer(
        start_lambda=CONFIG.klU_start,
        end_lambda=CONFIG.klU_end,
        n_steps=CONFIG.klU_steps)
elif CONFIG.klU_type == 'fixed':
    klU_annealer = FixedAnnealer(
        const_lambda=CONFIG.klU_const)
elif CONFIG.klU_type == 'none':
    klU_annealer = FixedAnnealer(
        const_lambda=0)
else:
    raise ValueError('Unknown annelear type.')

if __name__ == '__main__':
    print("Device in use: {}".format(CONFIG.DEVICE))

    model.train()
    model.to(CONFIG.DEVICE)

    iteration = 0
    for epoch in range(CONFIG.NUM_EPOCHS):
        for x, _ in train_loader:
            x = x.to(CONFIG.DEVICE)

            optimizer.zero_grad()

            temp = temp_annealer.step(iteration)
            klU_lambda = klU_annealer.step(iteration)

            x_recon, z_logits, z = model(x, temp)

            recon_loss = F.binary_cross_entropy(x_recon, x)
            kld_codes_loss = KLD_codes_uniform_loss(z)
            loss = recon_loss + klU_lambda * kld_codes_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                n_used_codes = len(z.detach().cpu().argmax(dim=1).view(-1).unique())
                print("Epoch: {} Iter: {} Loss: {} KL Loss (weighted): {} Recon Loss {} N codes used: {}".format(
                    epoch,
                    iteration,
                    round(loss.item(), 4),
                    round(kld_codes_loss.item(), 4),
                    round(recon_loss.item(), 4),
                    n_used_codes))

            writer.add_scalar('loss/recon_loss', recon_loss.item(), iteration)
            writer.add_scalar('loss/kld_codes_loss', kld_codes_loss.item(), iteration)
            writer.add_scalar('rates/temperature', temp, iteration)
            writer.add_scalar('rates/klU_lambda', klU_lambda, iteration)
            writer.add_scalar('additional_info/n_codes_used', n_used_codes, iteration)

            iteration += 1

        img_grid = torchvision.utils.make_grid(x[:8, :, :, :].detach().cpu())
        writer.add_image('images/original', img_grid, epoch)
        img_recon_grid = torchvision.utils.make_grid(x_recon[:8, :, :, :].detach().cpu())
        writer.add_image('images/reconstruction', img_recon_grid, epoch)

        lr_scheduler.step()

        model.save_model(CONFIG.save_model_path, CONFIG.save_model_name)




