import os
import re
from datasets.datasets.cub import CubDataset


class TextIndexer:
    def __init__(self):
        pass

    def collect_cub_vocab(self, root_text_path, imgs_list_file_path):
        dataset = CubDataset(
            root_img_path=None,
            root_text_path=root_text_path,
            imgs_list_file_path=imgs_list_file_path,
            transform=None,
            img_size=256,
            return_all_texts=True)

        texts_list = []
        for t in range(len(dataset)):
            texts_list += dataset.get_texts_only(t)
            # ' '.join(texts_list).replace('\n', '')
        pass


if __name__ == "__main__":
    ti = TextIndexer()
    ti.collect_cub_vocab(
        root_text_path="/home/andrey/Aalto/thesis/TA-VQVAE/data/CUB/text",
        imgs_list_file_path="/home/andrey/Aalto/thesis/TA-VQVAE/data/CUB/CUB_200_2011/images.txt"
    )


