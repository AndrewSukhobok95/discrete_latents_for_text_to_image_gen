import torch


def ng_quantize(z_logits):
    z = torch.zeros(z_logits.size(), device=z_logits.device)
    index = z_logits.argmax(dim=1)
    z = torch.scatter(z, 1, index.unsqueeze(dim=1), 1.0)
    return z

