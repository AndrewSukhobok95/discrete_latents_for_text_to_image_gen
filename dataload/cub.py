import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as torch_transforms


def _form_img_txt_paths(img_description):
    img_path = img_description.split()[1]
    txt_path = img_path.replace(".jpg", ".txt")
    return (img_path, txt_path)


def _read_imgs_list(imgs_list_file_path):
    with open(imgs_list_file_path) as f:
        data_list = f.readlines()
    data_list = list(map(_form_img_txt_paths, data_list))
    return data_list


def _parse_text(txt_path):
    with open(txt_path) as f:
        texts = f.readlines()
    return texts


class CubDataset(Dataset):
    def __init__(self, root_img_path, root_text_path, imgs_list_file_path=None, transform=None):
        self.root_img_path = root_img_path
        self.root_text_path = root_text_path
        if imgs_list_file_path is None:
            raise NotImplementedError("File ./CUB_200_2011/CUB_200_2011/images.txt is assumed.")
        else:
            self.data_paths = _read_imgs_list(imgs_list_file_path)
        if transform is None:
            transform = torch_transforms.Compose([
                torch_transforms.Resize(270),
                torch_transforms.RandomCrop(256),
                torch_transforms.RandomHorizontalFlip(),
                #torch_transforms.RandomRotation(5),
                torch_transforms.ToTensor()
            ])
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path, txt_path = self.data_paths[idx]
        img_full_path = os.path.join(self.root_img_path, img_path)
        txt_full_path = os.path.join(self.root_text_path, txt_path)

        img = Image.open(img_full_path)
        x = self.transform(img)
        x = torch.unsqueeze(x, 0)

        texts_list = _parse_text(txt_full_path)
        text_num = np.random.choice(len(texts_list))
        text = texts_list[text_num]

        # Remove later - used to check output
        # a = x.numpy().transpose((1,2,0))*255
        # a = a.astype(np.uint8)
        # _img = Image.fromarray(a, "RGB")
        # _img.show()
        # _img.save('my.png')

        return x, text


def cub_collate(samples):
    imgs, texts = list(zip(*samples))
    return torch.cat(imgs, dim=0), texts


if __name__ == '__main__':

    d = CubDataset(root_img_path="/home/andrey/Aalto/TA-VQVAE/data/CUB/CUB_200_2011/images",
                   root_text_path="/home/andrey/Aalto/TA-VQVAE/data/CUB/text",
                   imgs_list_file_path="/home/andrey/Aalto/TA-VQVAE/data/CUB/CUB_200_2011/images.txt")
    print(d[1])


