import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import argparse
from config_reader import ConfigReader
from modules.dvae.model import DVAE
from modules.transformer_gen.ar_cond_2stream.generator import LatentGenerator as LatentGenerator2s
from modules.transformer_gen.ar_cond_1stream.generator import LatentGenerator as LatentGenerator1s
from datasets.mnist_loader import MNISTData
from datasets.cub_loader import CUBData


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-cn', '--configname', action='store', type=str, required=True)
args = argument_parser.parse_args()

config_dir = '/home/andrey/Aalto/thesis/TA-VQVAE/configs/'
# config_dir = '/u/82/sukhoba1/unix/Desktop/TA-VQVAE/configs/'
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
        description_len=CONFIG.cond_seq_size,
        prct_train_split=0.99,
        custom_transform_version=CONFIG.custom_transform_version)
else:
    raise ValueError('Unknown dataset:', CONFIG.dataset)

train_loader = data_source.get_train_loader()


dvae = DVAE(
    in_channels=CONFIG.in_channels,
    vocab_size=CONFIG.vocab_size,
    num_x2downsamples=CONFIG.num_x2downsamples,
    num_resids_downsample=CONFIG.num_resids_downsample,
    num_resids_bottleneck=CONFIG.num_resids_bottleneck,
    hidden_dim=CONFIG.hidden_dim,
    device=CONFIG.DEVICE)

if CONFIG.model_type == '2s2s':
    G = LatentGenerator2s(
        hidden_width=CONFIG.hidden_width,
        hidden_height=CONFIG.hidden_height,
        embedding_dim=CONFIG.vocab_size,
        num_blocks=CONFIG.num_blocks,
        cond_num_blocks=CONFIG.cond_num_blocks,
        cond_seq_size=CONFIG.cond_seq_size,
        cond_vocab_size=CONFIG.cond_vocab_size,
        hidden_dim=CONFIG.hidden_dim,
        n_attn_heads=CONFIG.n_attn_heads,
        dropout_prob=CONFIG.dropout_prob,
        device=CONFIG.DEVICE)
elif CONFIG.model_type == '1s2s':
    G = LatentGenerator1s(
        hidden_width=CONFIG.hidden_width,
        hidden_height=CONFIG.hidden_height,
        embedding_dim=CONFIG.vocab_size,
        num_blocks=CONFIG.num_blocks,
        cond_seq_size=CONFIG.cond_seq_size,
        cond_vocab_size=CONFIG.cond_vocab_size,
        hidden_dim=CONFIG.hidden_dim,
        n_attn_heads=CONFIG.n_attn_heads,
        dropout_prob=CONFIG.dropout_prob,
        device=CONFIG.DEVICE)
else:
    raise ValueError('Unknown Generator type.')

dvae.eval()
G.train()

dvae.load_model(
    root_path=CONFIG.vae_model_path,
    model_name=CONFIG.vae_model_name)


print("Device in use: {}".format(CONFIG.DEVICE))

optimizer = optim.Adam(G.parameters(), lr=CONFIG.LR)
lr_scheduler = MultiStepLR(optimizer, milestones=CONFIG.step_LR_milestones, gamma=CONFIG.LR_gamma)
criteriation = nn.CrossEntropyLoss()

iteration = 0
for epoch in range(CONFIG.NUM_EPOCHS):
    for batch_index, (img, label) in enumerate(train_loader):
        img = img.to(CONFIG.DEVICE)
        label = label.permute(1, 0).to(CONFIG.DEVICE)

        with torch.no_grad():
            latent = dvae.ng_q_encode(img)

        b, emb, h, w = latent.size()
        x = latent.view(b, emb, -1).permute(2, 0, 1)

        start_vector = torch.zeros(1, b, emb, device=x.device)
        x_inp = torch.cat([start_vector, x[:-1,:,:]], dim=0)

        x_out = G(x_inp, label)

        seq_labels_pred = x_out.view(-1, emb)
        seq_lables_true = x.argmax(dim=2).view(-1)

        loss = criteriation(seq_labels_pred, seq_lables_true)
        loss.backward()

        if (batch_index + 1) % CONFIG.ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('loss/CrossEntropy', loss.item(), iteration)

            iteration += 1

            print("Epoch: {} Iter: {} Loss: {}".format(epoch, iteration, round(loss.item(), 5)))

    lr_scheduler.step()

    G.save_model(CONFIG.model_path, CONFIG.model_name)





