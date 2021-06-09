import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as torch_transforms


def _parse_obs(obs):
    values = obs.split('\n')
    desc = []
    for i in range(3):
        v = str(values[2 + i * 5].replace('+ value: ', ''))
        s = str(values[3 + i * 5].replace('+ size: ', ''))
        c = values[4 + i * 5].replace('+ color: ', '')
        p = values[5 + i * 5].replace('+ position: ', '')
        desc += [v, s, c, p]
    return values[0], desc


def parse_description_txt(path):
    with open(path, 'r') as f:
        data_str = f.read()
    obs_list = data_str.split('Obs ')[1:]
    obs_list = list(map(_parse_obs, obs_list))
    return dict(obs_list)


def parse_label_info_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


class MDMnistDataset(Dataset):
    def __init__(self,
                 root_data_path,
                 transforms=None):
        desc_path = os.path.join(root_data_path, "description/images_description.txt")
        lables_info_path = os.path.join(root_data_path, "description/labels_info.json")
        self.desc_dict = parse_description_txt(desc_path)
        self.lables_dict = parse_label_info_json(lables_info_path)

        self.imgs_path = os.path.join(root_data_path, "images")
        self.imgs = os.listdir(self.imgs_path)

        if transforms is None:
            transforms = torch_transforms.Compose([
                torch_transforms.ToTensor()
            ])
        self.transform = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        name = self.imgs[idx]
        img_path = os.path.join(self.imgs_path, name)
        name = name.replace('.png', '')

        img = Image.open(img_path).convert('RGB')
        x = self.transform(img)

        desc = self.desc_dict[name]
        labels = []
        for i in range(3):
            v_label = self.lables_dict['value_id_tree']['number'][desc[i * 4]]
            s_label = self.lables_dict['value_id_tree']['size'][desc[1 + i * 4]]
            c_label = self.lables_dict['value_id_tree']['color'][desc[2 + i * 4]]
            p_label = self.lables_dict['value_id_tree']['position'][desc[3 + i * 4]]
            labels += [v_label, s_label, c_label, p_label]

        return x, torch.LongTensor(labels)


if __name__ == '__main__':
    d = MDMnistDataset(root_data_path='/home/andrey/Aalto/thesis/TA-VQVAE/data/multi_descriptive_MNIST/')

    print(d[51])



