import torch
import torch.nn.functional as F
from torch import nn, optim

from modules.common_blocks import TrEncoderBlock
from modules.common_utils import subsequent_mask


class LatentGenerator(nn.Module):
    def __init__(self,
                 hidden_width,
                 hidden_height,
                 embedding_dim,
                 num_blocks,
                 hidden_dim,
                 n_attn_heads,
                 dropout_prob,
                 device=torch.device('cpu')):
        super(LatentGenerator, self).__init__()
        self.device = device

        self.hidden_width = hidden_width
        self.hidden_height = hidden_height
        self.embedding_dim = embedding_dim

        self.proj_in = nn.Linear(embedding_dim, embedding_dim)
        self.proj_out = nn.Linear(embedding_dim, embedding_dim)

        self.pe_col = nn.Parameter(torch.randn(hidden_width, 1, embedding_dim))
        self.pe_row = nn.Parameter(torch.randn(hidden_height, 1, embedding_dim))

        self.tr_encoder_blocks = nn.ModuleList([
            TrEncoderBlock(n_features=embedding_dim,
                           n_attn_heads=n_attn_heads,
                           n_hidden=hidden_dim,
                           dropout_prob=dropout_prob)
            for _ in range(num_blocks)
        ])

        self.to(self.device)

    def forward(self, x):
        seq_len, batch, emb = x.size()
        mask = subsequent_mask(seq_len).to(x.device)
        x = self.proj_in(x)

        pe_column = self.pe_col.repeat(self.hidden_width, 1, 1)
        pe_row = self.pe_row.repeat_interleave(self.hidden_height, dim=0)
        x = x + pe_column + pe_row

        for i, block in enumerate(self.tr_encoder_blocks):
            x = block(x, attn_mask=mask)
        x = self.proj_out(x)
        return x

    def sample(self, n_samples, return_start_token=False):
        seq_len = self.hidden_width * self.hidden_height
        samples = torch.zeros(seq_len + 1, n_samples, self.embedding_dim, device=self.device)

        for i in range(seq_len):
            out = self.forward(samples[:-1, :, :])
            probs = F.softmax(out[i, :, :], dim=-1)
            index = torch.multinomial(probs, num_samples=1)
            one_hot_sample = torch.zeros(n_samples, self.embedding_dim, device=self.device)
            one_hot_sample = torch.scatter(one_hot_sample, 1, index, 1.0)
            samples[i + 1, :, :] = one_hot_sample

        if return_start_token:
            return samples
        return samples[1:, :, :]



