import unittest
import torch

from modules.tadvae.discriminator.model import Discriminator


class TestTADVAEDiscriminatorBlocks(unittest.TestCase):

    def test_dim_Discriminator(self):
        x = torch.rand((2, 3, 256, 256))
        txt = torch.rand(10, 2, 32)
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        ])
        model = Discriminator(txt_in_dim=32)
        model.eval()
        f_cond, f_uncond = model.forward(x, txt, mask)
        self.assertEqual(f_cond.size(0), 2)
        self.assertEqual(f_uncond.size(0), 2)


if __name__ == '__main__':
    unittest.main()

