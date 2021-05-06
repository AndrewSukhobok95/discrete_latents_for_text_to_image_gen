import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSampleX2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, use_bn=True):
        super(DownSampleX2, self).__init__()
        self.use_bn = use_bn
        self._block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=bias),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self._block(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob=0.1):
        super(MLP, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(out_dim // 2, out_dim)
        )

    def forward(self, x):
        return self.block(x)


class EmbeddingMLP(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, out_dim, dropout_prob=0.1):
        super(EmbeddingMLP, self).__init__()

        self.emb = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim)

        self.mlp = MLP(
            in_dim=embedding_dim,
            out_dim=out_dim,
            dropout_prob=dropout_prob)

    def forward(self, x):
        x = self.emb(x)
        return self.mlp(x)


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


class TrDecoderBlock(nn.Module):
    def __init__(self, n_features, n_attn_heads=10, n_hidden=64, dropout_prob=0.1):
        """
        :param n_features:   Number of input and output features.
        :param n_attn_heads: Number of attention heads in the Multi-Head Attention.
        :param n_hidden:     Number of hidden units in the Feedforward (MLP) block.
        :param dropout_prob: Dropout rate after the first layer of the MLP and in
                             three places on the main path (before
                             combining the main path with a skip connection).
        """
        super(TrDecoderBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(n_features, n_attn_heads)
        self.ln1 = nn.LayerNorm(n_features)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.cross_attn = nn.MultiheadAttention(n_features, n_attn_heads)
        self.ln2 = nn.LayerNorm(n_features)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features)
        )
        self.ln3 = nn.LayerNorm(n_features)
        self.dropout3 = nn.Dropout(dropout_prob)

    def forward(self, tgt_seq, src_seq, src_pad_mask=None, attn_mask=None):
        """
        :param tgt_seq:      Transformed target sequences used as the inputs of the block.
        :type tgt_seq:         torch.tensor of shape (max_tgt_seq_length, batch_size, n_features)
        :param src_seq:      Encoded source sequences (outputs of the encoder).
        :type src_seq:         torch.tensor of shape (max_src_seq_length, batch_size, n_features)
        :param src_pad_mask: BoolTensor indicating which elements of the encoded source sequences should be ignored.
                             The values of True are ignored.
        :type src_pad_mask:    torch.BoolTensor of shape (batch_size, max_src_seq_length)
        :param attn_mask:    Subsequent mask to ignore subsequent elements of the target sequences in the inputs.
                             The rows of this matrix correspond to the output elements and the columns correspond
                             to the input elements.
        :type attn_mask:       torch.tensor of shape (max_tgt_seq_length, max_tgt_seq_length)
        :return:             Output tensor.
        :rtype:                torch.tensor of shape (max_seq_length, batch_size, n_features)
        Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
        """
        x = tgt_seq
        z = src_seq

        dx, _ = self.self_attn(query=x, key=x, value=x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + self.dropout1(dx))

        dx, _ = self.cross_attn(query=x, key=z, value=z, key_padding_mask=src_pad_mask, need_weights=False)
        x = self.ln2(x + self.dropout2(dx))

        dx = self.mlp(x)
        x = self.ln3(x + self.dropout3(dx))
        return x


class LatentGenerator(nn.Module):
    def __init__(self,
                 noise_dim,
                 hidden_height,
                 hidden_width,
                 hidden_vocab_size,
                 num_tr_blocks=8,
                 tr_hidden_dim=64,
                 tr_n_attn_heads=8,
                 dropout_prob=0.1):
        super(LatentGenerator, self).__init__()

        self.hidden_vocab_size = hidden_vocab_size
        self.hidden_height = hidden_height
        self.hidden_width = hidden_width
        self.num_z_positions = hidden_height * hidden_width
        self.num_cond_positions = 3

        self.tr_decoder_blocks = nn.ModuleList([
            TrDecoderBlock(n_features=hidden_vocab_size,
                           n_attn_heads=tr_n_attn_heads,
                           n_hidden=tr_hidden_dim,
                           dropout_prob=dropout_prob)
            for _ in range(num_tr_blocks)
        ])

        self.noise_mlp = MLP(
            in_dim=noise_dim,
            out_dim=hidden_vocab_size * self.num_z_positions,
            dropout_prob=dropout_prob)

        self.noise_pe = PositionalEmbedding(
            embedding_dim=hidden_vocab_size,
            num_embeddings=self.num_z_positions)

        self.cond_mlp = EmbeddingMLP(
            num_embeddings=10,
            embedding_dim=hidden_vocab_size,
            out_dim=hidden_vocab_size * self.num_cond_positions,
            dropout_prob=dropout_prob)

        self.cond_pe = PositionalEmbedding(
            embedding_dim=hidden_vocab_size,
            num_embeddings=self.num_cond_positions)

    def forward(self, noise, condition):
        noise_codes = self.noise_mlp(noise)
        noise_codes = noise_codes.view(-1, self.num_z_positions, self.hidden_vocab_size)
        noise_codes = self.noise_pe(noise_codes)
        x = noise_codes.permute(1, 0, 2)

        cond_codes = self.cond_mlp(condition)
        cond_codes = cond_codes.view(-1, self.num_cond_positions, self.hidden_vocab_size)
        cond_codes = self.cond_pe(cond_codes)
        cond_codes = cond_codes.permute(1, 0, 2)

        for i, block in enumerate(self.tr_decoder_blocks):
            x = block(x, cond_codes)

        z = x.permute(1, 2, 0).view(-1, self.hidden_vocab_size, self.hidden_width, self.hidden_height)

        return z






