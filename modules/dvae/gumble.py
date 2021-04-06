import torch
import torch.nn.functional as F


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, dim=-1):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return F.softmax(y / temperature, dim=dim)


def gumbel_softmax(logits, temperature, hard=False, dim=-1):
    y_soft = gumbel_softmax_sample(logits, temperature, dim=dim)
    if hard:
        ind = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(y_soft, device=y_soft.device).scatter_(dim, ind, 1.0)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y_soft).detach() + y_soft
        return y_hard
    else:
        return y_soft

