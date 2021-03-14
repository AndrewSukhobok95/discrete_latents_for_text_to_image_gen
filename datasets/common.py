import os
import numpy as np
from PIL import Image
import torch


def collate_fn(samples):
    imgs, texts = list(zip(*samples))
    return torch.cat(imgs, dim=0), texts


class Collater:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, samples):
        imgs, texts = list(zip(*samples))
        padded_sequences = self.tokenizer(texts, padding=True)
        token_tensor = torch.tensor(padded_sequences["input_ids"])
        token_type_tensor = torch.tensor(padded_sequences["token_type_ids"])
        attention_mask_tensor = torch.tensor(padded_sequences["attention_mask"])
        return torch.cat(imgs, dim=0), (token_tensor, token_type_tensor, attention_mask_tensor)