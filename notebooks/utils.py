import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show(img, figsize=(8, 4), plot_grid=False):
    img_cpu = img.detach().cpu()
    if plot_grid:
        b, ch, h, w = img_cpu.sizes()
        bg = torch.ones(b, ch, h+2, w+2) * 0.7
        bg[:, :, 1:h+1, 1:w+1] = img_cpu
        npimg = make_grid(bg).numpy()
    else:
        npimg = make_grid(img_cpu).numpy()
    plt.figure(figsize=figsize)
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()


