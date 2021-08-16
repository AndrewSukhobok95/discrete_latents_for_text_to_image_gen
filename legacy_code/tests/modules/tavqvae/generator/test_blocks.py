import unittest
import torch
from modules.tavqvae.generator.blocks import Attention, MaskBlock, masking_sum


class TestTAVQVAEBlocks(unittest.TestCase):

    def test_dim_Attention(self):
        x = torch.rand((2, 16, 5, 5))
        emb = torch.rand(10, 2, 16)
        mask = torch.tensor([
            [1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,0,0,0,0,0]
        ])
        model = Attention(embed_dim=16)
        model.eval()
        f = model.forward(x, emb, mask)
        self.assertEqual(x.size(), f.sizes())

    def test_dim_MaskBlock(self):
        x = torch.rand((2, 16, 5, 5))
        model = MaskBlock(ch_dim=16)
        model.eval()
        f = model.forward(x)
        expected_out_size = torch.Size([2, 1, 5, 5])
        self.assertEqual(f.sizes(), expected_out_size)

    def test_masking_sum(self):
        x1 = torch.ones((2, 4, 3, 3)) * 2
        x2 = torch.ones((2, 4, 3, 3)) * 3
        mask = torch.ones((2, 1, 3, 3))
        mask[0, :, 0, 0] = 0
        mask[0, :, 1, 1] = 0
        mask[1, :, 2, 0] = 0
        mask[1, :, 2, 2] = 0
        expected_result = torch.ones((2, 4, 3, 3)) * 2
        expected_result[0, :, 0, 0] = 3
        expected_result[0, :, 1, 1] = 3
        expected_result[1, :, 2, 0] = 3
        expected_result[1, :, 2, 2] = 3
        result = masking_sum(x1, x2, mask)
        self.assertTrue(torch.equal(result, expected_result))


if __name__ == '__main__':
    unittest.main()
