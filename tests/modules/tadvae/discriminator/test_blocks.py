import unittest
import torch

from modules.tadvae.discriminator.blocks import DownSampleX2


class TestTADVAEDiscriminatorBlocks(unittest.TestCase):

    def test_dim_DownSampleX2(self):
        x = torch.rand((2, 10, 8, 8))
        model = DownSampleX2(in_channels=10, out_channels=20)
        model.eval()
        f = model.forward(x)
        expected_output_size = torch.Size((2, 20, 4, 4))
        self.assertEqual(f.size(), expected_output_size)


if __name__ == '__main__':
    unittest.main()

