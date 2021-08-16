import unittest
import torch

from modules.generative_lsdvae.discriminator.model import Discriminator


class Test(unittest.TestCase):

    # def test_dim_Discriminator(self):
    #     x = torch.rand((2, 1, 28, 28))
    #     cond = torch.LongTensor([2, 5])
    #     model = Discriminator(in_channel=1, cond_in_dim=64, hidden_dim=128)
    #     model.eval()
    #     f = model.forward(x, cond)
    #     expected_output_size = torch.Size((2, 1, 28, 28))
    #     self.assertEqual(f.size(), expected_output_size)

    def test_dim_Discriminator(self):
        x = torch.rand((2, 1, 28, 28))
        model = Discriminator(in_channel=1)
        model.eval()
        f = model.forward(x)
        expected_output_size = torch.Size([2])
        self.assertEqual(f.size(), expected_output_size)


if __name__ == '__main__':
    unittest.main()

