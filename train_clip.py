import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from modules.dvae.model import DVAE
from modules.clip.model import CLIP, DVAECLIP
from config_reader import ConfigReader
from datasets.mnist_loader import MNISTData


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

data_source = MNISTData(
    img_type=CONFIG.dataset_type,
    root_path=CONFIG.root_path,
    batch_size=CONFIG.BATCH_SIZE)

train_loader = data_source.get_train_loader()


if CONFIG.use_vae:
    dvae = DVAE(
        in_channels=CONFIG.img_channels,
        vocab_size=CONFIG.vae_vocab_size,
        num_x2downsamples=CONFIG.vae_num_x2downsamples,
        num_resids_downsample=CONFIG.vae_num_resids_downsample,
        num_resids_bottleneck=CONFIG.vae_num_resids_bottleneck,
        hidden_dim=CONFIG.vae_hidden_dim,
        device=CONFIG.DEVICE)
    dvae.eval()
    dvae.load_model(
        root_path=CONFIG.vae_model_path,
        model_name=CONFIG.vae_model_name)


clip = CLIP(
    img_height=128,
    img_width=128,
    img_channels=3,
    patch_height=8,
    patch_width=8,
    txt_max_length=12,
    txt_vocab_size=20,
    embed_dim=128,
    num_blocks=8,
    hidden_dim=256,
    n_attn_heads=8,
    dropout_prob=0.1,
    device=CONFIG.DEVICE
)

optimizer = optim.Adam(clip.parameters(), lr=CONFIG.LR)
lr_scheduler = MultiStepLR(optimizer, milestones=CONFIG.step_LR_milestones, gamma=CONFIG.LR_gamma)

if __name__ == '__main__':
    print("Device in use: {}".format(CONFIG.DEVICE))

    clip.train()
    clip.to(CONFIG.DEVICE)

    iteration = 0
    for epoch in range(CONFIG.NUM_EPOCHS):
        for batch_index, (img, txt) in enumerate(train_loader):
            current_batch_size = img.size(0)

            img = img.to(CONFIG.DEVICE)
            txt = txt.permute(1, 0).to(CONFIG.DEVICE)
            labels = torch.arange(current_batch_size).to(CONFIG.DEVICE)

            if CONFIG.use_vae:
                with torch.no_grad():
                    img = dvae.get_reconstruction(img)

            logits_per_image, logits_per_text = clip(img, txt)

            loss_img = F.cross_entropy(logits_per_image, labels)
            loss_txt = F.cross_entropy(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2

            loss.backward()

            print("Epoch: {} Iter: {} Loss: {}".format(
                epoch, iteration, round(loss.item(), 4)))

            writer.add_scalar('loss/full', loss.item(), iteration)

            optimizer.step()
            optimizer.zero_grad()

            iteration += 1

        lr_scheduler.step()

        clip.save_model(CONFIG.save_model_path, CONFIG.save_model_name)



