import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets as torch_datasets
from torchvision import transforms as torch_transforms
from datasets.datasets.triple_mnist import TripleMnistDataset
from datasets.datasets.md_mnist import MDMnistDataset


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
    def __init__(self,
                 img_type,
                 root_path,
                 batch_size,
                 transforms=None,
                 custom_transform_version=None):
        types = ["classic", "56", "triple", "md"]

        self.img_type = img_type
        self.batch_size = batch_size

        if (transforms is None) and (custom_transform_version == 0):
            self.transforms = torch_transforms.Compose([
                torch_transforms.RandomRotation(10),
                torch_transforms.ToTensor()
            ])
        elif (transforms is None) and (custom_transform_version == 1):
            self.transforms = torch_transforms.Compose([
                torch_transforms.GaussianBlur(kernel_size=15),
                torch_transforms.ToTensor()
            ])
        else:
            self.transforms = torch_transforms.Compose([
                torch_transforms.ToTensor()
            ])

        if img_type in types[:2]:
            self.trainset = torch_datasets.MNIST(
                root=root_path, train=True, transform=self.transforms, download=False)
            self.testset = torch_datasets.MNIST(
                root=root_path, train=False, transform=self.transforms, download=False)
        elif img_type == types[2]:
            self.trainset = TripleMnistDataset(
                root_img_path=root_path, transforms=transforms)
        elif img_type == types[3]:
            self.trainset = MDMnistDataset(
                root_data_path=root_path, transforms=self.transforms)
        else:
            raise ValueError("Choose one of the following types:" + str(types))

        if img_type == types[1]:
            self.collate_fn = collate_mnist_56
        else:
            self.collate_fn = None

    def get_train_loader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        loader = DataLoader(
            dataset=self.trainset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn)
        return loader

    def get_test_loader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        loader = DataLoader(
            dataset=self.testset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn)
        return loader







