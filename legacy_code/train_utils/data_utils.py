import torch
from torch.utils.data import DataLoader
from datasets.common import collate_fn, Collater

from datasets.cub import CubDataset


def get_cub_dataloaders(tokenizer,
                        root_img_path,
                        root_text_path,
                        imgs_list_file_path,
                        img_size,
                        batch_size,
                        prct_train_split=0.95,
                        seed=42):

    dataset = CubDataset(root_img_path=root_img_path,
                         root_text_path=root_text_path,
                         imgs_list_file_path=imgs_list_file_path,
                         img_size=img_size)

    train_length = int(len(dataset) * prct_train_split)
    test_length = len(dataset) - train_length
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, lengths=[train_length, test_length], generator=torch.Generator().manual_seed(seed))

    collater = Collater(tokenizer=tokenizer)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collater.collate_fn)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collater.collate_fn)

    return train_loader, test_loader