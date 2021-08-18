import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import argparse
from config_reader import ConfigReader
from utilities.model_loading import *
from datasets.mnist_loader import MNISTData
from datasets.cub_loader import CUBData


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
        description_len=CONFIG.cond_seq_size,
        prct_train_split=0.999,
        custom_transform_version=CONFIG.custom_transform_version)
else:
    raise ValueError('Unknown dataset:', CONFIG.dataset)

train_loader = data_source.get_train_loader()


dvae = define_DVAE(CONFIG, eval=True, load=True, compound_config=True)

_load_to_continue, _load = False, False
if hasattr(CONFIG, 'load_model_path') and hasattr(CONFIG, 'load_model_name'):
    _load_to_continue, _load = True, True

if CONFIG.model_type == '2s2s':
    G = define_LatentGenerator2s(CONFIG, eval=False, load=_load, load_to_continue=_load_to_continue)
elif CONFIG.model_type == '1s2s':
    G = define_LatentGenerator1s(CONFIG, eval=False, load=_load, load_to_continue=_load_to_continue)
else:
    raise ValueError('Unknown Generator type.')

print("Device in use: {}".format(CONFIG.DEVICE))

optimizer = optim.Adam(G.parameters(), lr=CONFIG.LR)

if hasattr(CONFIG, 'LR_gamma') and hasattr(CONFIG, 'step_LR_milestones'):
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

    if hasattr(CONFIG, 'LR_gamma') and hasattr(CONFIG, 'step_LR_milestones'):
        lr_scheduler.step()

    G.save_model(CONFIG.save_model_path, CONFIG.save_model_name)





