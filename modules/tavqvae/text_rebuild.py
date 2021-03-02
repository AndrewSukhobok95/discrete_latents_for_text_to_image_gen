import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.tavqvae.blocks import ResidualStack


class TextRebuildBlock(nn.Module):
    def __init__(self):
        super(TextRebuildBlock, self).__init__()


