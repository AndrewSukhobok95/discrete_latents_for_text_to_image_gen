import os
import torch
from torch import nn
import torch.nn.functional as F
from modules.common_blocks import TrDecoderBlock
from modules.common_utils import subsequent_mask


class Generator(nn.Module):
    def __init__(self,
                 hidden_width,
                 hidden_height,
                 embedding_dim,
                 num_blocks,
                 hidden_dim,
                 n_attn_heads,
                 dropout_prob):
        super(Generator, self).__init__()

        self.hidden_width = hidden_width
        self.hidden_height = hidden_height
        self.embedding_dim = embedding_dim

        self.proj_in = nn.Linear(embedding_dim, embedding_dim)
        self.proj_out = nn.Linear(embedding_dim, embedding_dim)

        self.pe_col = nn.Parameter(torch.randn(hidden_width, 1, embedding_dim))
        self.pe_row = nn.Parameter(torch.randn(hidden_height, 1, embedding_dim))

        self.tr_dec_blocks = nn.ModuleList([
            TrDecoderBlock(
                n_features=embedding_dim,
                n_attn_heads=n_attn_heads,
                n_hidden=hidden_dim,
                dropout_prob=dropout_prob)
            for _ in range(num_blocks)
        ])

    def forward(self, x, noise):
        seq_len, batch, emb = x.size()
        mask = subsequent_mask(seq_len).to(x.device)
        x = self.proj_in(x)

        pe_column = self.pe_col.repeat(self.hidden_width, 1, 1)
        pe_row = self.pe_row.repeat_interleave(self.hidden_height, dim=0)
        x = x + pe_column + pe_row

        for i, block in enumerate(self.tr_dec_blocks):
            x, _, _ = block(x, noise, attn_mask=mask)

        x = self.proj_out(x)
        return x

    def sample(self, n_samples, device, noise=None, return_start_token=False):
        noise_dim = (1, n_samples, self.embedding_dim)
        if noise is None:
            noise = torch.randn(noise_dim, device=device)
        else:
            assert noise.size() == noise_dim, "Wrong dimension, must be " + str(noise_dim)
            assert noise.device == device, "Noise must be on the provided device: " + str(device)

        seq_len = self.hidden_width * self.hidden_height
        samples = torch.zeros(seq_len + 1, n_samples, self.embedding_dim, device=device)
        for i in range(seq_len):
            out = self.forward(samples[:-1, :, :], noise)
            probs = F.softmax(out[i, :, :], dim=-1)
            index = torch.multinomial(probs, num_samples=1)
            one_hot_sample = torch.zeros(n_samples, self.embedding_dim, device=device)
            one_hot_sample = torch.scatter(one_hot_sample, 1, index, 1.0)
            samples[i + 1, :, :] = one_hot_sample

        if return_start_token:
            return samples
        return samples[1:, :, :]

    def sample_from(self, start_seq, start_index, noise, device):
        return

    def save_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        path = os.path.join(root_path, model_name + ".pth")
        torch.save(self.state_dict(), path)

    def load_model(self, root_path, model_name, map_location=torch.device('cpu')):
        path = os.path.join(root_path, model_name + ".pth")
        self.load_state_dict(torch.load(path, map_location=map_location))




