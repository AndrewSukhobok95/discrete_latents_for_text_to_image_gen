import unittest
import torch

from modules.dtr_gan.generator import Generator


class Test(unittest.TestCase):

    def test_dim_Generator(self):
        noise = torch.rand((2, 100))
        model = Generator(
            noise_dim=100,
            hidden_width=7,
            hidden_height=7,
            embedding_dim=10,
            num_blocks=5,
            n_attn_heads=5,
            hidden_dim=64,
            dropout_prob=0.1,
            num_latent_positions=7*7)
        model.eval()
        f = model.forward(noise)
        expected_output_size = torch.Size([2, 10, 7, 7])
        self.assertEqual(f.size(), expected_output_size)


if __name__ == '__main__':
    unittest.main()