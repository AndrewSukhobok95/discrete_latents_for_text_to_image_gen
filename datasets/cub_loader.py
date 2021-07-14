import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.datasets.cub import CubDataset
from cub_text_indexer import TextIndexer


class TextCollater:
    def __init__(self, vocab_file_path):
        self.text_indexer = TextIndexer(vocab_file_path=vocab_file_path)

    def collate_fn(self, samples):
        imgs, texts = list(zip(*samples))
        texts_ids = []
        for t in texts:
            texts_ids.append(self.text_indexer.text2ids(t))
        return torch.stack(imgs), texts


class CUBData:
    def __init__(self,
                 img_type,
                 root_path,
                 batch_size,
                 prct_train_split=0.95,
                 transforms=None,
                 seed=None):
        types = ["128_text", "256_text", "128_token_ids", "256_token_ids"]

        self.img_type = img_type
        self.batch_size = batch_size

        root_img_path = os.path.join(root_path, "CUB_200_2011/images")
        root_text_path = os.path.join(root_path, "text")
        imgs_list_file_path = os.path.join(root_path, "CUB_200_2011/images.txt")
        vocab_file_path = os.path.join(root_path, "vocab.json")

        if 'token_ids' in img_type:
            text_collater = TextCollater(vocab_file_path=vocab_file_path)

        if img_type == types[0]:
            img_size = 128
            self.collate_fn = None
        elif img_type == types[1]:
            img_size = 256
            self.collate_fn = None
        elif img_type == types[2]:
            img_size = 128
            self.collate_fn = text_collater.collate_fn
        elif img_type == types[3]:
            img_size = 256
            self.collate_fn = text_collater.collate_fn
        else:
            raise ValueError("Choose one of the following types:" + str(types))

        dataset = CubDataset(root_img_path=root_img_path,
                             root_text_path=root_text_path,
                             imgs_list_file_path=imgs_list_file_path,
                             img_size=img_size,
                             transforms=transforms)

        self.train_length = int(len(dataset) * prct_train_split)
        self.test_length = len(dataset) - self.train_length

        self.trainset, self.testset = torch.utils.data.random_split(
            dataset, lengths=[self.train_length, self.test_length],
            generator=torch.Generator().manual_seed(seed) if seed else None)

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


if __name__=="__main__":
    cubdata = CUBData(
        img_type="128_token_ids",#"128_token_ids"
        root_path="/home/andrey/Aalto/thesis/TA-VQVAE/data/CUB",
        batch_size=4)

    loader = cubdata.get_train_loader()

    img, text = next(iter(loader))

    pass




