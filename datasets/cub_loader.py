import os
import torch
from torch.utils.data import DataLoader
from datasets.common import Collater

from datasets.datasets.cub import CubDataset


class CUBData:
    def __init__(self,
                 img_type,
                 root_path,
                 batch_size,
                 prct_train_split=0.95,
                 seed=None):
        types = ["128_text", "256_text"]

        self.img_type = img_type
        self.batch_size = batch_size

        if img_type == types[0]:
            img_size = 128
        elif img_type == types[1]:
            img_size = 256
        else:
            raise ValueError("Choose one of the following types:" + str(types))

        root_img_path = os.path.join(root_path, "CUB_200_2011/images")
        root_text_path = os.path.join(root_path, "text")
        imgs_list_file_path = os.path.join(root_path, "CUB_200_2011/images.txt")

        dataset = CubDataset(root_img_path=root_img_path,
                             root_text_path=root_text_path,
                             imgs_list_file_path=imgs_list_file_path,
                             img_size=img_size)

        self.train_length = int(len(dataset) * prct_train_split)
        self.test_length = len(dataset) - self.train_length

        self.trainset, self.testset = torch.utils.data.random_split(
            dataset, lengths=[self.train_length, self.test_length],
            generator=torch.Generator().manual_seed(seed) if seed else None)

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




