import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_embeddings=1024):
        super(PositionalEmbedding, self).__init__()
        self.pe = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.register_buffer("position_ids", torch.arange(num_embeddings).expand((1, -1)))

    def forward(self, x):
        """
        :param x: Flatten image representation
        :type x:   torch.tensor of shape (batch x seq_len x embedding_dim)
        :return:  Flatten image representation with positional embedding
        :rtype:    torch.tensor of shape (batch x seq_len x embedding_dim)
        """
        position_embeddings = self.pe(self.position_ids)
        return x + position_embeddings


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob=0.1, bias=False):
        super(MLP, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=bias),
            nn.ReLU(),  # nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(out_dim, out_dim, bias=bias)
        )

    def forward(self, x):
        return self.block(x)



