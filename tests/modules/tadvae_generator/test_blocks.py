import unittest
import torch

from modules.tadvae_generator.blocks import LearnablePositionalImageEmbedding, ImageEmbeddingReweighter


class TestTADVAEBlocks(unittest.TestCase):

    def test_dim_EmbeddingReweighter(self):
        x = torch.rand((2, 10, 4, 4))
        model = ImageEmbeddingReweighter(in_dim=10, out_dim=20)
        model.eval()
        f = model.forward(x)
        expected_output_size = torch.Size((2, 20, 4, 4))
        self.assertEqual(f.size(), expected_output_size)

    def test_dim_LearnablePositionalEmbedding(self):
        x = torch.rand((2, 10, 4, 4))
        model = LearnablePositionalImageEmbedding(embedding_dim=10, n_positions=16)
        model.eval()
        f = model.forward(x)
        self.assertEqual(x.size(), f.size())


if __name__ == '__main__':
    unittest.main()

