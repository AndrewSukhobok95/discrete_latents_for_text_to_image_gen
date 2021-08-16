import unittest
import torch
from modules.tavqvae.generator.text_rebuild_block import TextRebuildBlock


class TestTAVQVAETextRebuildBlock(unittest.TestCase):

    def test_dim_TextRebuildBlock(self):
        x = torch.rand((2, 16, 5, 5))
        emb = torch.rand(2, 10, 32)
        mask = torch.tensor([
            [1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,0,0,0,0,0]
        ])
        model = TextRebuildBlock(channel_dim=16, embed_dim=32, num_residual_layers=5)
        model.eval()
        f = model.forward(x, emb, mask)
        self.assertEqual(x.size(), f.size())

