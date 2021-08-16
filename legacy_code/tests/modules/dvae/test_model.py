import unittest
import torch
from modules.dvae.model import DVAE


class TestDVAE(unittest.TestCase):

    def test_dim_DVAE(self):
        x = torch.rand((2, 3, 128, 128))
        model = DVAE(in_channels=3,
                     vocab_size=16,
                     num_x2downsamples=2,
                     num_resids_downsample=2,
                     num_resids_bottleneck=2)
        model.eval()
        f = model.forward(x)
        self.assertEqual(x.size(), f.size())


if __name__ == '__main__':
    unittest.main()

