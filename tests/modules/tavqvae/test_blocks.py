import unittest
import torch
from modules.tavqvae.blocks import Attention


class TestTAVQVAEBlocks(unittest.TestCase):

    def test_dim_Residual(self):
        x = torch.rand((2, 16, 5, 5))
        emb = torch.rand(2, 10, 32)
        model = Attention(embed_dim=32, text_embed_dim=16)
        model.eval()
        f = model.forward(x, emb)
        # self.assertEqual(x.size(), f.size())


if __name__ == '__main__':
    unittest.main()
