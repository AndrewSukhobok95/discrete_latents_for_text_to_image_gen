import torch.nn as nn
from modules.tadvae.generator.blocks import TrDecoderBlock, ImageEmbeddingReweighter,\
    LearnablePositionalImageEmbedding, MaskBlock, ConvSuperPixelPredictor, masking_sum


class TextRebuildBlock(nn.Module):
    def __init__(self,
                 img_hidden_dim,
                 txt_hidden_dim,
                 n_trd_blocks=4,
                 num_trd_block_for_mask=3,
                 n_attn_heads=4,
                 linear_hidden_dim=1024,
                 dropout_prob=0.1,
                 n_img_hidden_positions=1024):
        super(TextRebuildBlock, self).__init__()

        assert n_trd_blocks >= num_trd_block_for_mask, "Number of blocks used for mask prediction " \
                                                       "must be less than the general number of " \
                                                       "Transformer Decoder blocks"

        self.num_trd_block_for_mask = num_trd_block_for_mask

        self.img_emb = ImageEmbeddingReweighter(in_dim=img_hidden_dim, out_dim=txt_hidden_dim)
        self.img_pos_emb = LearnablePositionalImageEmbedding(embedding_dim=txt_hidden_dim, n_positions=n_img_hidden_positions)
        self.tr_decoder_blocks = nn.ModuleList([
            TrDecoderBlock(txt_hidden_dim, n_attn_heads, linear_hidden_dim, dropout_prob)
            for _ in range(n_trd_blocks)
        ])
        self.mask_block = MaskBlock(ch_dim=txt_hidden_dim, n_resid_blocks=2, bias=True, hidden_dim=txt_hidden_dim//2)
        self.logits_predictor = ConvSuperPixelPredictor(in_ch=txt_hidden_dim, out_ch=img_hidden_dim,
                                                        n_resid_blocks=2, bias=True, hidden_dim=txt_hidden_dim//2)

    def forward(self, img_h, txt_h, txt_pad_mask):
        """
        :param img_h:        Image representation.
        :type img_h:           torch.tensor of shape (batch x ch x h x w)
        :param txt_h:        Text representation.
        :type txt_h:           torch.tensor of shape (seq_len x batch x emb_size)
        :param txt_pad_mask: Text Mask with 0 elements indicating which elements to ignore.
        :type txt_pad_mask:    torch.tensor of shape (batch x seq_len)
        :return:             Image representation corrected by text.
        :rtype:                torch.tensor of shape (batch x ch x h x w)
        """
        txt_pad_mask = (1 - txt_pad_mask).bool()

        img_e = self.img_emb(img_h)
        img_e = self.img_pos_emb(img_e)

        eb, ech, eh, ew = img_e.sizes()
        x = img_e.view(eb, ech, eh*ew).permute(2, 0, 1)

        mask_img_h = None
        for i, block in enumerate(self.tr_decoder_blocks):
            x = block(x, txt_h, txt_pad_mask)
            if i+1 == self.num_trd_block_for_mask:
                mask_x = x.permute(1, 2, 0).view(eb, ech, eh, ew)
                mask_img_h = self.mask_block(mask_x)

        x = x.permute(1, 2, 0).view(eb, ech, eh, ew)
        img_h_new = self.logits_predictor(x)

        return masking_sum(img_h_new, img_h, mask_img_h), mask_img_h

