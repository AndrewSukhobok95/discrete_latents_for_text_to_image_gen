import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.common_blocks import ResidualStack, ChangeChannels


class ImageEmbeddingReweighter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ImageEmbeddingReweighter, self).__init__()
        self.block = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Linear(in_dim, out_dim, bias=False)
        )

    def forward(self, img_h):
        """
        :param img_h: Image representation
        :type img_h:   torch.tensor of shape (batch x ch x h x w)
        :return:      Weighted representation
        :rtype:        torch.tensor of shape (batch x ch x h x w)
        """
        img_reshaped = img_h.permute(0, 2, 3, 1)  # -> (batch x h x w x ch)
        emb = self.block(img_reshaped).permute(0, 3, 1, 2)
        return emb


class LearnablePositionalImageEmbedding(nn.Module):
    def __init__(self, embedding_dim, n_positions=1024):
        super(LearnablePositionalImageEmbedding, self).__init__()
        self.pe = nn.Embedding(n_positions, embedding_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(n_positions).expand((1, -1)),
        )

    def forward(self, x):
        """
        :param x: Image representation
        :type x:   torch.tensor of shape (batch x ch x h x w)
        :return:  Image representation with positional embedding
        :rtype:    torch.tensor of shape (batch x ch x h x w)
        """
        b, ch, h, w = x.sizes()
        position_embeddings = self.pe(self.position_ids)
        pe_reshaped = position_embeddings.repeat(b, 1, 1).permute(0, 2, 1).view(b, ch, h, w)
        return x + pe_reshaped


class TrDecoderBlock(nn.Module):
    def __init__(self, n_features, n_attn_heads, n_hidden=64, dropout_prob=0.1):
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

    def forward(self, tgt_seq, src_seq, src_pad_mask, attn_mask=None):
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


class MaskBlock(nn.Module):
    def __init__(self, ch_dim, n_resid_blocks=2, bias=True, hidden_dim=None):
        super(MaskBlock, self).__init__()

        if hidden_dim is None:
            hidden_dim = ch_dim

        self._block = nn.Sequential(
            nn.Conv2d(in_channels=ch_dim, out_channels=hidden_dim, kernel_size=1, stride=1, bias=bias),
            ResidualStack(hidden_dim, hidden_dim, n_resid_blocks, bias=bias, use_bn=False),
            ChangeChannels(in_channels=hidden_dim, out_channels=1, bias=bias, use_bn=False)
        )

    def forward(self, x):
        """
        :param x: Image representation.
        :type x:   torch.tensor of shape (batch x ch x h x w)
        :return:  Mask indicating which elements to change in x.
        :rtype:    torch.tensor of shape (batch x 1 x h x w)
        """
        x = self._block(x)
        return F.hardsigmoid(x)


class ConvSuperPixelPredictor(nn.Module):
    def __init__(self, in_ch, out_ch, n_resid_blocks=2, bias=True, hidden_dim=None):
        super(ConvSuperPixelPredictor, self).__init__()

        if hidden_dim is None:
            hidden_dim = in_ch

        self._block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=hidden_dim, kernel_size=1, stride=1, bias=bias),
            ResidualStack(hidden_dim, hidden_dim, n_resid_blocks, bias=bias, use_bn=False),
            ChangeChannels(in_channels=hidden_dim, out_channels=out_ch, bias=bias, use_bn=False)
        )

    def forward(self, x):
        """
        :param x: Image representation.
        :type x:   torch.tensor of shape (batch x in_ch x h x w)
        :return:  Image representation with out_ch to be the number of classes.
        :rtype:    torch.tensor of shape (batch x out_ch x h x w)
        """
        x = self._block(x)
        return x


def masking_sum(x_new, x_old, mask):
    """
    :param x_new: tensor from which we want to take mask positions marked by 1
    :type x_new:   torch.tensor of shape (batch x ch x h x w)
    :param x_old: tensor from which we want to take mask positions marked by 0
    :type x_old:   torch.tensor of shape (batch x ch x h x w)
    :param mask:  tensor indicating which elements to take from x_new and x_old
    :type mask:    torch.tensor of shape (batch x 1 x h x w)
    :return:      tensor that combines x_new and x_old with mask
    :rtype:        torch.tensor of shape (batch x ch x h x w)
    """
    mask_ch = mask.repeat((1, x_new.sizes(1), 1, 1))
    masked_x1 = x_new * mask_ch
    masked_x2 = x_old * (1 - mask_ch)
    return masked_x1 + masked_x2

