import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets as torch_datasets
from torchvision import transforms as torch_transforms
from datasets.datasets.triple_mnist import TripleMnistDataset


def collate_mnist_56(data_list):
    imgs, lables = zip(*data_list)
    imgs = torch.stack(imgs, dim=0)
    lables = torch.LongTensor(lables)
    b, ch, img_h, img_w = imgs.size()
    img_base_h = img_h * 2
    img_base_w = img_w * 2
    img_base = torch.zeros(b, ch, img_base_h, img_base_w)
    coords_list = [(0, 0), (0, 28), (28, 0), (28, 28), (14, 14)]
    for i in range(b):
        index = np.random.randint(len(coords_list))
        x1, y1 = coords_list[index]
        x2 = x1 + img_w
        y2 = y1 + img_h
        img_base[i, :, y1:y2, x1:x2] = imgs[i, :, :, :]
    return img_base, lables


class MNISTData:
    def __init__(self, img_type, root_path, batch_size, transforms=None):
        mnist_types = ["classic", "classic_56", "triple"]

        self.img_type = img_type
        self.batch_size = batch_size

        if transforms is None:
            self.transforms = torch_transforms.Compose([
                torch_transforms.RandomRotation(10),
                torch_transforms.ToTensor()
            ])

        if img_type in mnist_types[:2]:
            self.trainset = torch_datasets.MNIST(
                root=root_path, train=True, transform=self.transforms, download=False)
        elif img_type == mnist_types[2]:
            self.trainset = TripleMnistDataset(
                root_img_path=root_path, transforms=transforms)
        else:
            raise ValueError("Choose one of the following types:" + str(mnist_types))

        if img_type == mnist_types[1]:
            self.collate_fn = collate_mnist_56
        else:
            self.collate_fn = None

    def get_train_loader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.trainset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        return train_loader




