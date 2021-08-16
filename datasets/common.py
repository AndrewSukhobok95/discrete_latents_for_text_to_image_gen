import os
import numpy as np
import torch
import torchvision.transforms.functional as TF


def collate_fn(samples):
    imgs, texts = list(zip(*samples))
    return torch.cat(imgs, dim=0), texts


class SquarePad:
    def __call__(self, image):
        w, h = image.sizes
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return TF.pad(image, padding, 0, 'symmetric')




