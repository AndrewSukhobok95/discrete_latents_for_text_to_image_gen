import os
import torch
from torch import nn
import torch.nn.functional as F

from modules.common_utils import latent_to_img


def one_step_pg_loss(seq, noise, G, G_rollout, D, dvae, start_index,
                     n_rollouts, hidden_height, hidden_width, device):
    s_seq_len, n_samples, embedding_dim = seq.size()
    n_iters = s_seq_len - start_index

    n_rollout_samples = n_samples * n_rollouts

    out = G.forward(seq[:-1, :, :], noise)
    log_P = F.log_softmax(out[start_index, :, :], dim=-1)

    with torch.no_grad():

        rollout_samples = torch.zeros(s_seq_len, n_samples, embedding_dim, device=device)
        rollout_samples[:start_index + 1, :, :] = seq[:start_index + 1, :, :]

        rollout_noise = noise.repeat_interleave(n_rollouts, dim=1)
        rollout_samples = rollout_samples.repeat_interleave(n_rollouts, dim=1)

        for i in range(n_iters - 1):
            cur_index = start_index + i

            out = G_rollout.forward(rollout_samples[:-1, :, :], rollout_noise)
            probs = F.softmax(out[cur_index, :, :], dim=-1)

            index = torch.multinomial(probs, num_samples=1)
            one_hot_sample = torch.zeros(n_rollout_samples, embedding_dim, device=device)
            one_hot_sample = torch.scatter(one_hot_sample, 1, index, 1.0)
            rollout_samples[cur_index + 1, :, :] = one_hot_sample

        gen_rollout_samples = latent_to_img(rollout_samples[1:, :, :], dvae, hidden_height, hidden_width)

        rewards = D(gen_rollout_samples).view(n_samples, n_rollouts).mean(dim=1)

    loss = 0

    tgt_index = seq[start_index + 1, :, :].argmax(dim=-1).detach().cpu().numpy()

    for i in range(n_samples):
        loss += - log_P[i, tgt_index[i]] * rewards[i]

    return loss



