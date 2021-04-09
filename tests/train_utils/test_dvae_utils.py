import unittest
import torch
import torch.nn.functional as F

from train_utils.dvae_utils import KLD_uniform_loss


class TestDVAETrainUtils(unittest.TestCase):

    def test_KLD_uniform_loss(self):
        z_logits_uniform = torch.rand((4, 10, 5, 5))
        z_dist_uniform = F.softmax(
            z_logits_uniform,
            dim=1
        )
        kld_loss_uniform = KLD_uniform_loss(z_dist_uniform)

        z_logits_fixed = torch.rand((4, 10, 5, 5))
        z_logits_fixed[:, 0, :, :] = 10
        z_dist_fixed = F.softmax(
            z_logits_fixed,
            dim=1
        )
        kld_loss_fixed = KLD_uniform_loss(z_dist_fixed)

        self.assertGreater(kld_loss_fixed.item(), kld_loss_uniform.item())


if __name__ == '__main__':
    unittest.main()

