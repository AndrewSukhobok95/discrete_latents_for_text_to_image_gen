import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from collections import Counter
import matplotlib.pyplot as plt

from utilities.model_loading import *
from config_reader import ConfigReader
from utilities.md_mnist_utils import DescriptionGenerator
from utilities.md_mnist_utils import LabelsInfo
from utilities.utils import EvalReport
from modules.common_utils import latent_to_img, img_to_latent


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-cn', '--configname', action='store', type=str, required=True)
args = argument_parser.parse_args()

# config_dir = '/home/andrey/dev/TA-VQVAE/configs/'
config_dir = '/u/82/sukhoba1/unix/Desktop/TA-VQVAE/configs/'
config_name = args.configname
config_path = os.path.join(config_dir, config_name)

CONFIG = ConfigReader(config_path=config_path)
CONFIG.print_config_info()

CONFIG_G1 = ConfigReader(config_path=CONFIG.generator_1_config_path)
CONFIG_G2 = ConfigReader(config_path=CONFIG.generator_2_config_path)
CONFIG_clip = ConfigReader(config_path=CONFIG.clip_config_path)

_eval = True
_load = True
dvae = define_DVAE(CONFIG_G1, eval=_eval, load=_load, compound_config=True)
G1s = define_LatentGenerator1s(CONFIG_G1, eval=_eval, load=_load)
G2s = define_LatentGenerator2s(CONFIG_G2, eval=_eval, load=_load)
clip = define_CLIP(CONFIG_clip, eval=_eval, load=_load)

description_gen = DescriptionGenerator(batch_size=CONFIG.batch_size)
labels_info = LabelsInfo(json_path=CONFIG.labels_info_path)

eval_report = EvalReport(
    root_path=CONFIG.report_root_path,
    report_name=CONFIG.report_name)


def batch_cics_calculation(x1, x2, t, clip):
    n_obs = t.size(1)

    x = torch.cat([x1, x2], dim=0)
    logits_per_image, logits_per_text = clip(x, t)

    l1 = torch.diagonal(logits_per_image[:n_obs, :])
    l2 = torch.diagonal(logits_per_image[n_obs:, :])
    probs = F.softmax(torch.stack([l1, l2], dim=1), dim=1)

    n2 = probs.argmax(dim=1).sum().item()
    n1 = n_obs - n2
    return n1, n2


def batch_ctrs_calculation(x1, x2, t, clip):
    n_obs = t.size(1) // 13

    _, logits_txt_1 = clip(x1, t)
    _, logits_txt_2 = clip(x2, t)

    l1 = torch.empty(n_obs, 13)
    l2 = torch.empty(n_obs, 13)
    for i in range(n_obs):
        s = 13 * i
        e = 13 * i + 13
        l1[i, :] = F.softmax(logits_txt_1[s:e, i], dim=0)
        l2[i, :] = F.softmax(logits_txt_2[s:e, i], dim=0)

    d1 = dict(Counter(l1.argmax(dim=1).tolist()))
    d2 = dict(Counter(l2.argmax(dim=1).tolist()))
    return d1, d2


if __name__=="__main__":
    print("Device in use: {}".format(CONFIG.DEVICE))

    for _ in range(CONFIG.num_iterations):
        txt = description_gen.sample_with_modifications()
        x_txt = torch.LongTensor(labels_info.encode_values(txt))
        x_txt = x_txt.permute(1, 0).to(CONFIG.DEVICE)

        n_obs = x_txt.size(1) // 13
        true_index = [i * 13 for i in range(n_obs)]
        x_txt_true = x_txt[:, true_index].to(CONFIG.DEVICE)

        with torch.no_grad():
            print('+ G1 generation')
            x_img_g1 = G1s.sample(x_txt_true)
            x_img_g1 = latent_to_img(x_img_g1, dvae, CONFIG_G1.hidden_height, CONFIG_G1.hidden_width)

            print('+ G2 generation')
            x_img_g2 = G2s.sample(x_txt_true)
            x_img_g2 = latent_to_img(x_img_g2, dvae, CONFIG_G2.hidden_height, CONFIG_G2.hidden_width)

        # x_img_g1 = torch.rand((4, 3, 128, 128))
        # x_img_g2 = torch.rand((4, 3, 128, 128))

        print('+ CICS calculation')
        s1, s2 = batch_cics_calculation(x_img_g1, x_img_g2, x_txt_true, clip)
        eval_report.update_cics(score_1=s1, score_2=s2)

        print('+ CTRS calculation')
        d1, d2 = batch_ctrs_calculation(x_img_g1, x_img_g2, x_txt, clip)
        eval_report.update_ctrs(d1=d1, d2=d2)

        print('G1 score: {} G2 score: {}'.format(*eval_report.get_cics()))
        print('G1 peak: {} G2 peak: {}'.format(*eval_report.get_ctrs_peaks()))

        eval_report.save()

        print()




