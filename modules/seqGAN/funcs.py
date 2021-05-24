import torch


def latent_to_img(sample, dvae, hidden_height, heddin_width):
    seq_len, batch, emb = sample.size()
    latent = sample.permute(1, 2, 0).view(batch, -1, hidden_height, heddin_width)
    img = dvae.ng_q_decode(latent)
    return img