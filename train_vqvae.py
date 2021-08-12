import os
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
from config_reader import ConfigReader
from datasets.mnist_loader import MNISTData
from datasets.cub_loader import CUBData
from modules.vqvae.model import VQVAE
from utilities.vqvae_utils import AdamOptWithMultiStepLrVQVAE

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
        img_type=CONFIG.dataset_type,
        root_path=CONFIG.root_path,
        batch_size=CONFIG.BATCH_SIZE)
elif CONFIG.dataset == 'cub':
    data_source = CUBData(
        img_type=CONFIG.dataset_type,
        root_path=CONFIG.root_path,
        batch_size=CONFIG.BATCH_SIZE,
        prct_train_split=0.99)

train_loader = data_source.get_train_loader()


model = VQVAE(img_channels=CONFIG.in_channels,
              num_embeddings=CONFIG.vocab_size,
              embedding_dim=CONFIG.vqvae_embedding_dim,
              commitment_cost=CONFIG.vqvae_commitment_cost,
              decay=CONFIG.vqvae_decay,
              num_x2downsamples=CONFIG.num_x2downsamples,
              num_resid_downsample_layers=CONFIG.num_resids_downsample,
              num_resid_bottleneck_layers=CONFIG.num_resids_bottleneck,
              use_batch_norm=True,
              use_conv1x1=True)

model.train()
model.to(CONFIG.DEVICE)

optimizer = AdamOptWithMultiStepLrVQVAE(
    type=CONFIG.OPT_type,
    model=model,
    lr=CONFIG.LR,
    lr_q=CONFIG.LR_gamma,
    milestones=CONFIG.step_LR_milestones,
    gamma=CONFIG.LR_gamma)

if __name__ == '__main__':
    print("Device in use: {}".format(CONFIG.DEVICE))

    iteration = 0
    for epoch in range(CONFIG.NUM_EPOCHS):
        for imgs, _ in train_loader:
            imgs = imgs.to(CONFIG.DEVICE)

            vq_loss, quantized, data_recon, perplexity, n_codes_used = model(imgs)

            recon_error = F.mse_loss(data_recon, imgs)
            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            print("Epoch: {} Iter: {} Loss: {}".format(epoch, iteration, loss.item()))

            writer.add_scalar('loss/full', loss.item(), iteration)
            writer.add_scalar('loss/vq_loss', vq_loss.item(), iteration)
            writer.add_scalar('loss/recon_loss', recon_error.item(), iteration)
            writer.add_scalar('rates/perplexity', perplexity.item(), iteration)
            writer.add_scalar('additional_info/n_codes_used', n_codes_used, iteration)

            iteration += 1

        optimizer.lr_step()

        model.save_model(root_path=CONFIG.save_model_path, model_name=CONFIG.save_model_name)

        img_grid = torchvision.utils.make_grid(imgs[:8, :, :, :].detach().cpu())
        writer.add_image('images/original', img_grid, epoch)

        img_recon_grid = torchvision.utils.make_grid(data_recon[:8, :, :, :].detach().cpu())
        writer.add_image('images/reconstruction', img_recon_grid, epoch)

    writer.close()
