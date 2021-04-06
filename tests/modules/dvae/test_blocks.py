import unittest
import torch
from modules.dvae.blocks import DownSampleX2, UpSampleX2, Encoder, Decoder


class TestDVAEBlocks(unittest.TestCase):

    def test_dim_DownSampleX2(self):
        x = torch.rand((1, 3, 128, 128))
        model = DownSampleX2(in_channels=3, out_channels=16)
        expected_out_size = torch.Size([1, 16, 64, 64])
        model.eval()
        f = model.forward(x)
        self.assertEqual(expected_out_size, f.size())

    def test_dim_UpSampleX2(self):
        x = torch.rand((1, 16, 64, 64))
        model = UpSampleX2(in_channels=16, out_channels=3)
        expected_out_size = torch.Size([1, 3, 128, 128])
        model.eval()
        f = model.forward(x)
        self.assertEqual(expected_out_size, f.size())

    def test_dim_Encoder(self):
        x = torch.rand((1, 3, 128, 128))
        model = Encoder(in_channels=3, out_channels=20, num_x2downsamples=2)
        expected_out_size = torch.Size([1, 20, 32, 32])
        model.eval()
        f = model.forward(x)
        self.assertEqual(expected_out_size, f.size())

    def test_dim_Decoder(self):
        x = torch.rand((1, 256, 32, 32))
        model = Decoder(in_channels=256, out_channels=3, num_x2upsamples=2)
        expected_out_size = torch.Size([1, 3, 128, 128])
        model.eval()
        f = model.forward(x)
        self.assertEqual(expected_out_size, f.size())


if __name__ == '__main__':
    unittest.main()

