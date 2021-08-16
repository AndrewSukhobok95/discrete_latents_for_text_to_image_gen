import unittest
import torch
from modules.tavqvae.discriminator.tagan_model import Discriminator


class TestTAGANDiscriminator(unittest.TestCase):

    def test_dim_Discriminator(self):
        x = torch.rand((2, 3, 128, 128))
        emb = torch.rand(10, 2, 16)
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        ])
        model = Discriminator(text_embedding_dim=16)
        model.eval()
        disc, sim, sim_n = model.forward(img=x, txt=emb, len_txt=mask.sum(dim=1), negative=True)
        expected_disc_size = torch.Size([2])
        expected_sim_size = torch.Size([2])
        expected_sim_n_size = torch.Size([2])
        self.assertEqual(expected_disc_size, disc.sizes())
        self.assertEqual(expected_sim_size, sim.sizes())
        self.assertEqual(expected_sim_n_size, sim_n.sizes())


if __name__ == '__main__':
    unittest.main()
