import os
import torch
import torch.nn.functional as F
from torch import nn

from modules.common_blocks import TrDecoderBlock
from modules.common_utils import subsequent_mask
from modules.transformer_gen.ar_cond_2stream.cond_encoder import CondEncoder


class LatentGenerator(nn.Module):
    def __init__(self,
                 hidden_width,
                 hidden_height,
                 embedding_dim,
                 num_blocks,
                 cond_num_blocks,
                 cond_seq_size,
                 cond_vocab_size,
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

        self.tr_decoder_blocks = nn.ModuleList([
            TrDecoderBlock(n_features=embedding_dim,
                           n_attn_heads=n_attn_heads,
                           n_hidden=hidden_dim,
                           dropout_prob=dropout_prob)
            for _ in range(num_blocks)
        ])

        self.cond_encoder = CondEncoder(
            cond_seq_size=cond_seq_size,
            cond_vocab_size=cond_vocab_size,
            embedding_dim=embedding_dim,
            num_blocks=cond_num_blocks,
            hidden_dim=hidden_dim,
            n_attn_heads=n_attn_heads,
            dropout_prob=dropout_prob,
            device=self.device
        )

        self.to(self.device)

    def forward(self, x, condition):
        seq_len, batch, emb = x.size()
        mask = subsequent_mask(seq_len).to(x.device)
        x = self.proj_in(x)

        pe_column = self.pe_col.repeat(self.hidden_width, 1, 1)
        pe_row = self.pe_row.repeat_interleave(self.hidden_height, dim=0)
        x = x + pe_column + pe_row

        c = self.cond_encoder(condition)

        for i, block in enumerate(self.tr_decoder_blocks):
            x, _, _ = block(x, c, attn_mask=mask)

        x = self.proj_out(x)
        return x

    def sample(self, condition, return_start_token=False):
        '''
        :param condition: torch.LongTensor of size (seq_len x batch)
        '''
        _, n_samples = condition.size()
        seq_len = self.hidden_width * self.hidden_height
        samples = torch.zeros(seq_len + 1, n_samples, self.embedding_dim, device=self.device)

        with torch.no_grad():
            for i in range(seq_len):
                out = self.forward(samples[:-1, :, :], condition)
                probs = F.softmax(out[i, :, :], dim=-1)
                index = torch.multinomial(probs, num_samples=1)
                one_hot_sample = torch.zeros(n_samples, self.embedding_dim, device=self.device)
                one_hot_sample = torch.scatter(one_hot_sample, 1, index, 1.0)
                samples[i + 1, :, :] = one_hot_sample

        if return_start_token:
            return samples
        return samples[1:, :, :]

    def save_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        path = os.path.join(root_path, model_name + ".pth")
        torch.save(self.state_dict(), path)

    def load_model(self, root_path, model_name, map_location=None):
        if map_location is None:
            map_location = self.device
        path = os.path.join(root_path, model_name + ".pth")
        self.load_state_dict(torch.load(path, map_location=map_location))




