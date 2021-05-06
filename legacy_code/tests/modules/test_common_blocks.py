import unittest
import torch
from modules.common_blocks import Residual, ResidualStack, ChangeChannels


class TestCommonBlocks(unittest.TestCase):

    def test_dim_Residual(self):
        x = torch.rand((1, 3, 128, 128))
        model = Residual(in_channels=3, out_channels=3)
        model.eval()
        f = model.forward(x)
        self.assertEqual(x.size(), f.size())

    def test_dim_ResidualStack(self):
        x = torch.rand((1, 3, 128, 128))
        model = ResidualStack(in_channels=3, out_channels=3, num_residual_layers=10)
        model.eval()
        f = model.forward(x)
        self.assertEqual(x.size(), f.size())

    def test_dim_ChangeChannels(self):
        x = torch.rand((1, 10, 128, 128))
        model = ChangeChannels(in_channels=10, out_channels=3, use_bn=True)
        model.eval()
        f = model.forward(x)
        expected_out_size = torch.Size([1, 3, 128, 128])
        self.assertEqual(expected_out_size, f.size())


if __name__ == '__main__':
    unittest.main()