import torch


def label_like(label, x):
    assert label == 0 or label == 1
    v = torch.zeros_like(x) if label == 0 else torch.ones_like(x)
    v = v.to(x.device)
    return v


def zeros_like(x):
    return label_like(0, x)


def ones_like(x):
    return label_like(1, x)


