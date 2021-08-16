import unittest
import torch

from modules.generative_lsdvae.blocks import LatentGenerator


class Test(unittest.TestCase):

    def test_dim_LatentGenerator(self):
        noise = torch.rand((2, 10))
        cond = torch.LongTensor([2, 5])
        model = LatentGenerator(
            noise_dim=10,
            hidden_height=7,
            hidden_width=7,
            hidden_vocab_size=16,
            num_tr_blocks=8,
            tr_hidden_dim=64,
            tr_n_attn_heads=8)
        model.eval()
        f = model.forward(noise, cond)
        expected_output_size = torch.Size((2, 16, 7, 7))
        self.assertEqual(f.sizes(), expected_output_size)


if __name__ == '__main__':
    unittest.main()

