import unittest
import torch

from modules.dtr_gan.discriminator import Discriminator


class Test(unittest.TestCase):

    def test_dim_Discriminator(self):
        x = torch.rand((2, 10, 7, 7))
        model = Discriminator(
            embedding_dim=10,
            num_blocks=5,
            n_attn_heads=5,
            hidden_dim=64,
            dropout_prob=0.1,
            num_latent_positions=7*7+1)
        model.eval()
        f = model.forward(x)
        expected_output_size = torch.Size([2])
        self.assertEqual(f.size(), expected_output_size)


if __name__ == '__main__':
    unittest.main()

