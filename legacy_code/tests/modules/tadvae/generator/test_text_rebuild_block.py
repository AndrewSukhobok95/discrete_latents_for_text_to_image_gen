import unittest
import torch

from modules.tadvae.generator.text_rebuild_block import TextRebuildBlock


class TestTextRebuildBlock(unittest.TestCase):

    def test_dim_EmbeddingReweighter(self):
        x = torch.rand((2, 8, 5, 5))
        z = torch.rand(10, 2, 16)
        mask_z = (1 - torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        ])).bool()
        model = TextRebuildBlock(img_hidden_dim=8,
                                 txt_hidden_dim=16,
                                 n_trd_blocks=4,
                                 n_attn_heads=4,
                                 linear_hidden_dim=32,
                                 dropout_prob=0.1,
                                 n_img_hidden_positions=5 * 5)
        model.eval()
        f = model.forward(x, z, mask_z)
        self.assertEqual(f.sizes(), x.size())


if __name__ == '__main__':
    unittest.main()

