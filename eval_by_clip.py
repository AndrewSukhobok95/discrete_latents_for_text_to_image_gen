import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
import argparse

from modules.dvae.model import DVAE
from modules.clip.model import CLIP, DVAECLIP
from modules.transformer_gen.ar_cond_1stream.generator import LatentGenerator as LatentGenerator1s
from modules.transformer_gen.ar_cond_2stream.generator import LatentGenerator as LatentGenerator2s
from config_reader import ConfigReader
from datasets.mnist_loader import MNISTData
from utilities.md_mnist_utils import DescriptionGenerator
from utilities.md_mnist_utils import LabelsInfo
from utilities.utils import EvalReport
from modules.common_utils import latent_to_img, img_to_latent


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-cn', '--configname', action='store', type=str, required=True)
args = argument_parser.parse_args()

config_dir = '/u/82/sukhoba1/unix/Desktop/TA-VQVAE/configs/'
config_name = args.configname
config_path = os.path.join(config_dir, config_name)

CONFIG = ConfigReader(config_path=config_path)
CONFIG.print_config_info()

CONFIG_G1 = ConfigReader(config_path=CONFIG.generator_1_config_path)
CONFIG_G2 = ConfigReader(config_path=CONFIG.generator_2_config_path)
CONFIG_clip = ConfigReader(config_path=CONFIG.clip_config_path)


dvae = DVAE(
    in_channels=CONFIG_G1.in_channels,
    vocab_size=CONFIG_G1.vocab_size,
    num_x2downsamples=CONFIG_G1.num_x2downsamples,
    num_resids_downsample=CONFIG_G1.num_resids_downsample,
    num_resids_bottleneck=CONFIG_G1.num_resids_bottleneck,
    hidden_dim=CONFIG_G1.hidden_dim,
    device=CONFIG_G1.DEVICE)

dvae.eval()
dvae.load_model(
    root_path=CONFIG_G1.vae_model_path,
    model_name=CONFIG_G1.vae_model_name)

G1s = LatentGenerator1s(
    hidden_width=CONFIG_G1.hidden_width,
    hidden_height=CONFIG_G1.hidden_height,
    embedding_dim=CONFIG_G1.vocab_size,
    num_blocks=CONFIG_G1.num_blocks,
    cond_seq_size=CONFIG_G1.cond_seq_size,
    cond_vocab_size=CONFIG_G1.cond_vocab_size,
    hidden_dim=CONFIG_G1.hidden_dim,
    n_attn_heads=CONFIG_G1.n_attn_heads,
    dropout_prob=CONFIG_G1.dropout_prob,
    device=CONFIG_G1.DEVICE)

G1s.eval()
G1s.load_model(
    root_path=CONFIG_G1.model_path,
    model_name=CONFIG_G1.model_name)

G2s = LatentGenerator2s(
    hidden_width=CONFIG_G2.hidden_width,
    hidden_height=CONFIG_G2.hidden_height,
    embedding_dim=CONFIG_G2.vocab_size,
    num_blocks=CONFIG_G2.num_blocks,
    cond_num_blocks=CONFIG_G2.cond_num_blocks,
    cond_seq_size=CONFIG_G2.cond_seq_size,
    cond_vocab_size=CONFIG_G2.cond_vocab_size,
    hidden_dim=CONFIG_G2.hidden_dim,
    n_attn_heads=CONFIG_G2.n_attn_heads,
    dropout_prob=CONFIG_G2.dropout_prob,
    device=CONFIG_G2.DEVICE)

G2s.eval()
G2s.load_model(
    root_path=CONFIG_G2.model_path,
    model_name=CONFIG_G2.model_name)

clip = CLIP(
    img_height=CONFIG_clip.img_height,
    img_width=CONFIG_clip.img_width,
    img_channels=CONFIG_clip.img_channels,
    patch_height=CONFIG_clip.patch_height,
    patch_width=CONFIG_clip.patch_width,
    txt_max_length=CONFIG_clip.txt_max_length,
    txt_vocab_size=CONFIG_clip.txt_vocab_size,
    embed_dim=CONFIG_clip.embed_dim,
    num_blocks=CONFIG_clip.num_blocks,
    hidden_dim=CONFIG_clip.hidden_dim,
    n_attn_heads=CONFIG_clip.n_attn_heads,
    dropout_prob=CONFIG_clip.dropout_prob,
    device=CONFIG_clip.DEVICE)

clip.eval()
clip.load_model(
    root_path=CONFIG_clip.save_model_path,
    model_name=CONFIG_clip.save_model_name)


description_gen = DescriptionGenerator(batch_size=CONFIG.batch_size)
labels_info = LabelsInfo(json_path=CONFIG.labels_info_path)

eval_report = EvalReport(
    root_path=CONFIG.report_root_path,
    report_name=CONFIG.report_name)

if __name__=="__main__":
    print("Device in use: {}".format(CONFIG.DEVICE))

    g1_score = 0
    g2_score = 0

    for _ in range(CONFIG.num_iterations):
        txt = description_gen.sample()

        x_txt = x_txt.permute(1, 0).to(CONFIG.DEVICE)

        n_obs = x_txt.sizes(1)

        with torch.no_grad():
            x_img_g1 = G1s.sample(x_txt)
            x_img_g1 = latent_to_img(x_img_g1, dvae, CONFIG_G1.hidden_height, CONFIG_G1.hidden_width)

            x_img_g2 = G2s.sample(x_txt)
            x_img_g2 = latent_to_img(x_img_g2, dvae, CONFIG_G2.hidden_height, CONFIG_G2.hidden_width)

            x = torch.cat([x_img_g1, x_img_g2], dim=0)
            logits_per_image, logits_per_text = clip(x, x_txt)

            l1 = torch.diagonal(logits_per_image[:n_obs, :])
            l2 = torch.diagonal(logits_per_image[n_obs:, :])

            probs = F.softmax(torch.stack([l1, l2], dim=1), dim=1)

        n2 = probs.argmax(dim=1).sum().item()
        n1 = n_obs - n2

        g1_score += n1
        g2_score += n2

        print('G1 score: {} G2 score: {}'.format(g1_score, g2_score))

    eval_report.save(g1_score, g2_score)




