import torch


def subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def latent_to_img(sample, dvae, hidden_height, heddin_width):
    seq_len, batch, emb = sample.size()
    latent = sample.permute(1, 2, 0).view(batch, -1, hidden_height, heddin_width)
    img = dvae.ng_q_decode(latent)
    return img


