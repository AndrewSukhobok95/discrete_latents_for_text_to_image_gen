import torch


def subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def latent_to_img(sample, dvae, img_latent_height, img_latent_width):
    seq_len, batch, emb = sample.size()
    latent = sample.permute(1, 2, 0).view(batch, -1, img_latent_height, img_latent_width)
    img = dvae.ng_q_decode(latent)
    return img


def img_to_latent(sample, dvae):
    with torch.no_grad():
        latent = dvae.ng_q_encode(sample)
    b, emb, h, w = latent.size()
    x = latent.view(b, emb, -1).permute(2, 0, 1)
    return x


