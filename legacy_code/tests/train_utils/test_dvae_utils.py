import unittest
import torch
import torch.nn.functional as F

from train_utils.dvae_utils import KLD_uniform_loss


class TestDVAETrainUtils(unittest.TestCase):

    def test_KLD_uniform_loss(self):
        z_logits_uniform = torch.ones((4, 10, 5, 5)) / 10
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

        def next_num(max_num):
            num = -1
            while True:
                if num == max_num:
                    num = -1
                num += 1
                yield num

        z_logits_every = torch.zeros((4, 10, 5, 5))
        g = next_num(9)
        for i in range(5):
            for j in range(5):
                z_logits_every[:, next(g), i, j] = 1
        # z_dist_every = F.softmax(
        #     z_logits_every,
        #     dim=1
        # )
        # kld_loss_every = KLD_uniform_loss(z_dist_every)

        self.assertGreater(kld_loss_fixed.item(), kld_loss_uniform.item())


if __name__ == '__main__':
    unittest.main()

