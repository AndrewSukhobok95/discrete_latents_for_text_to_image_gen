import unittest
import torch

from modules.generative_lsdvae.generator.model import Generator


class Test(unittest.TestCase):

    def test_dim_LatentGenerator(self):
        noise = torch.rand((2, 10))
        cond = torch.LongTensor([2, 5])
        model = Generator(
            noise_dim=10,
            hidden_height=7,
            hidden_width=7,
            hidden_vocab_size=16,
            out_channels=1,
            num_tr_blocks=8,
            dvae_num_x2upsamples=2,
            dvae_num_resids_upsample=3,
            dvae_num_resids_bottleneck=4,
            dvae_hidden_dim=64,
            tr_hidden_dim=64,
            tr_n_attn_heads=8,
            dropout_prob=0.1)
        model.eval()
        f = model.forward(noise, cond)
        expected_output_size = torch.Size((2, 1, 28, 28))
        self.assertEqual(f.size(), expected_output_size)



if __name__ == '__main__':
    unittest.main()

