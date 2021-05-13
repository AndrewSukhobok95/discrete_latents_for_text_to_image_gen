import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as torch_transforms


def _join_paths(path1, path2):
    return os.path.abspath(os.path.normpath(os.path.join(path1, path2)))


def _recusive_collect_files_paths(root_path):
    files = []
    for file in os.listdir(root_path):
        cur_path = _join_paths(root_path, file)
        if os.path.isdir(cur_path):
            files_from_level = _recusive_collect_files_paths(cur_path)
            files = files + files_from_level
        else:
            files.append(cur_path)
    return files


def get_data_paths(root_path):
    summary_file_name = "data_paths.txt"
    summary_file_paths = _join_paths(root_path, summary_file_name)
    if os.path.exists(summary_file_paths):
        with open(summary_file_paths, "r") as f:
            data_list = f.readlines()
        data_list = list(map(lambda s: s.replace("\n", ""), data_list))
    else:
        data_list = _recusive_collect_files_paths(root_path)
        with open(summary_file_paths, "w") as f:
            f.write("\n".join(data_list))
        print("File with data paths is created at {}.".format(summary_file_paths))
    return data_list


class TripleMnistDataset(Dataset):
    def __init__(self,
                 root_img_path,
                 transforms=None):
        train_path = os.path.join(root_img_path, "train")
        self.data_paths = get_data_paths(train_path)
        if transforms is None:
            transforms = torch_transforms.Compose([
                torch_transforms.RandomRotation(5),
                torch_transforms.ToTensor()
            ])
        self.transform = transforms

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]

        img = Image.open(img_path)
        x = self.transform(img)

        digit = os.path.dirname(img_path).split("/")[-1]
        labels = list(map(int, digit))

        return x, torch.LongTensor(labels)


if __name__ == '__main__':
    # files = _recusive_collect_files_paths(root_path="/home/andrey/Aalto/thesis/TA-VQVAE/data/tripleMNIST/train")

    d = TripleMnistDataset(root_img_path="/data/tripleMNIST/train")
    print(d[51])

    print()

