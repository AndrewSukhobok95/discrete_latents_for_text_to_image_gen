import os
import torch
import yaml


def recursive_read_dict(d):
    built_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_d = recursive_read_dict(v)
            built_d.update(new_d)
        else:
            built_d.update({k: v})
    return built_d


class ConfigReader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            info = yaml.safe_load(file)
        info_flat = recursive_read_dict(info)
        for k, v in info_flat.items():
            self.__setattr__(k, v)

    def print_config_info(self):
        row_format ="{:<40}" * 2
        config_params = vars(self)
        keys = sorted(config_params.keys())
        for k in keys:
            print(row_format.format(k, str(config_params[k])))


if __name__ == '__main__':
    c = ConfigReader(config_path="/home/andrey/Aalto/thesis/TA-VQVAE/configs/tadvae_cub_local.yaml")
    c.print_config_info()
