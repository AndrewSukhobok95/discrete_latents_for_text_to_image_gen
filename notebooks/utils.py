import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show(img, figsize=(8, 4)):
    img_grid = make_grid(img.detach().cpu())
    plt.figure(figsize=figsize)
    npimg = img_grid.numpy()
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()


