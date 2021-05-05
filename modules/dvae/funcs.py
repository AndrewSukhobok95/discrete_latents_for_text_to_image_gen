import torch


def ng_quantize(z_logits, dim=1):
    z = torch.zeros(z_logits.size(), device=z_logits.device)
    index = z_logits.argmax(dim=dim)
    z = torch.scatter(z, dim, index.unsqueeze(dim=dim), 1.0)
    return z

