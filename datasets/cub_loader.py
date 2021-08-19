import os
import numpy as np
import torch
from torchvision import transforms as torch_transforms
from torch.utils.data import DataLoader
from datasets.datasets.cub import CubDataset
from datasets.cub_text_indexer import TextIndexer
from datasets.common import SquarePad


class TextCollater:
    def __init__(self, vocab_file_path, description_len, pad_value=0):
        self.text_indexer = TextIndexer(vocab_file_path=vocab_file_path)
        self.pad_value = pad_value
        self.description_len = description_len

    def collate_fn(self, samples):
        imgs, texts = list(zip(*samples))
        texts_ids = []
        for t in texts:
            texts_ids.append(self.text_indexer.text2ids(t))
        texts_ids_array = self.pad_value * np.ones((len(texts_ids), self.description_len))
        for i, ti in enumerate(texts_ids):
            texts_ids_array[i, :len(ti)] = ti
        return torch.stack(imgs), torch.tensor(texts_ids_array, dtype=torch.long)


class CUBData:
    def __init__(self,
                 img_type,
                 root_path,
                 batch_size,
                 description_len=None,
                 prct_train_split=0.95,
                 transforms=None,
                 custom_transform_version=None,
                 seed=None):
        types = ["32_text", "64_text", "128_text", "256_text",
                 "32_token_ids", "64_token_ids", "128_token_ids", "256_token_ids"]

        self.img_type = img_type
        self.batch_size = batch_size

        root_img_path = os.path.join(root_path, "CUB_200_2011/images")
        root_text_path = os.path.join(root_path, "text")
        imgs_list_file_path = os.path.join(root_path, "CUB_200_2011/images.txt")
        vocab_file_path = os.path.join(root_path, "vocab.json")

        if 'token_ids' in img_type:
            text_collater = TextCollater(
                vocab_file_path=vocab_file_path,
                description_len=description_len)

        if img_type == types[0]:
            img_size = 32
            self.collate_fn = None
        elif img_type == types[1]:
            img_size = 64
            self.collate_fn = None
        elif img_type == types[2]:
            img_size = 128
            self.collate_fn = None
        elif img_type == types[3]:
            img_size = 256
            self.collate_fn = None
        elif img_type == types[4]:
            img_size = 32
            self.collate_fn = text_collater.collate_fn
        elif img_type == types[5]:
            img_size = 64
            self.collate_fn = text_collater.collate_fn
        elif img_type == types[6]:
            img_size = 128
            self.collate_fn = text_collater.collate_fn
        elif img_type == types[7]:
            img_size = 256
            self.collate_fn = text_collater.collate_fn
        else:
            raise ValueError("Choose one of the following types:" + str(types))

        if (transforms is None) and (custom_transform_version == 0):
            transforms = torch_transforms.Compose([
                torch_transforms.Resize(img_size + 10),
                torch_transforms.RandomRotation(2),
                torch_transforms.RandomCrop(img_size),
                torch_transforms.RandomHorizontalFlip(),
                torch_transforms.ToTensor()
            ])
        elif (transforms is None) and (custom_transform_version == 1):
            transforms = torch_transforms.Compose([
                SquarePad(),
                torch_transforms.RandomRotation(2),
                torch_transforms.Resize(img_size + 20),
                torch_transforms.CenterCrop(img_size),
                torch_transforms.RandomHorizontalFlip(),
                torch_transforms.ToTensor()
            ])

        self.dataset = CubDataset(root_img_path=root_img_path,
                             root_text_path=root_text_path,
                             imgs_list_file_path=imgs_list_file_path,
                             img_size=img_size,
                             transforms=transforms)

        self.train_length = int(len(self.dataset) * prct_train_split)
        self.test_length = len(self.dataset) - self.train_length

        self.trainset, self.testset = torch.utils.data.random_split(
            self.dataset, lengths=[self.train_length, self.test_length],
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

    def get_imgs_by_file_names(self, imgs_paths, img_size=128):
        transforms = torch_transforms.Compose([
            SquarePad(),
            torch_transforms.Resize(img_size + 20),
            torch_transforms.CenterCrop(img_size)
        ])
        data_paths = list(map(lambda x: x[0], self.dataset.data_paths))
        indices = []
        for ip in imgs_paths:
            indices.append(data_paths.index(ip))
        out = torch.empty(len(indices), 3, img_size, img_size)
        for i, idx in enumerate(indices):
            out[i, :, :, :] = transforms(self.dataset[idx][0])
        return out


if __name__=="__main__":
    cubdata = CUBData(
        img_type="128_token_ids",#"128_token_ids"
        root_path="/home/andrey/dev/TA-VQVAE/data/CUB",
        batch_size=4,
        description_len=64)

    cubdata.get_imgs_by_file_names([
        '001.Black_footed_Albatross/Black_Footed_Albatross_0008_796083.jpg',
        '009.Brewer_Blackbird/Brewer_Blackbird_0009_2616.jpg'
    ])

    loader = cubdata.get_train_loader()

    img, text = next(iter(loader))

    pass




