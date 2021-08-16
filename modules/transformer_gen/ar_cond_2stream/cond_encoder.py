import os
import torch
import torch.nn.functional as F
from torch import nn

from modules.common_blocks import TrEncoderBlock


class CondEncoder(nn.Module):
    def __init__(self,
                 cond_seq_size,
                 cond_vocab_size,
                 embedding_dim,
                 num_blocks,
                 hidden_dim,
                 n_attn_heads,
                 dropout_prob,
                 device=torch.device('cpu')):
        super(CondEncoder, self).__init__()
        self.device = device

        self.cond_embedding = nn.Embedding(
            num_embeddings=cond_vocab_size,
            embedding_dim=embedding_dim)
        self.pe = nn.Parameter(torch.randn(cond_seq_size, 1, embedding_dim))

        self.tr_encoder_blocks = nn.ModuleList([
            TrEncoderBlock(n_features=embedding_dim,
                           n_attn_heads=n_attn_heads,
                           n_hidden=hidden_dim,
                           dropout_prob=dropout_prob)
            for _ in range(num_blocks)
        ])

        self.to(self.device)

    def forward(self, x):
        '''
        :param x: torch.LongTensor of size (seq_len x batch)
        '''
        _, batch = x.sizes()
        x = self.cond_embedding(x)
        x = x + self.pe.repeat(1, batch, 1)
        for i, block in enumerate(self.tr_encoder_blocks):
            x = block(x)
        return x



