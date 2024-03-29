import numpy as np
import torch
import torch.nn.functional as F


def KLD_uniform_loss(z_logits):
    eps = 1e-20
    b, lat_dim, h, w = z_logits.size()
    log_prior = torch.log(torch.ones((b, lat_dim, h * w), device=z_logits.device) / lat_dim)

    z_probs = F.softmax(z_logits, dim=1)

    z_dist_flatten = z_probs.view(b, lat_dim, -1)
    log_z_dist = torch.log(z_dist_flatten + eps)

    KLD = torch.sum(z_dist_flatten * (log_z_dist - log_prior), dim=[1, 2]).mean()
    return KLD


def KLD_codes_uniform_loss(z):
    eps = 1e-20
    b, lat_dim, h, w = z.size()
    N = h * w

    log_prior = torch.log(torch.ones((b, lat_dim), device=z.device) / lat_dim)

    z_dist_flatten = z.sum(dim=[2, 3]) / N
    log_z_dist = torch.log(z_dist_flatten + eps)

    KLD = torch.sum(z_dist_flatten * (log_z_dist - log_prior), dim=1).mean()
    return KLD


class TemperatureAnnealer:
    def __init__(self, start_temp=1, end_temp=1/16, n_steps=100000):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.n_steps = n_steps
        self.k = (end_temp - start_temp) / (n_steps - 1)
        self.b = start_temp - self.k

    def step(self, step):
        if step == 0:
            return self.start_temp
        elif step > self.n_steps:
            return self.end_temp
        else:
            return self.k * step + self.b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class KLDWeightAnnealer:
    def __init__(self, start_lambda=0, end_lambda=5, n_steps=5000):
        self.start_lambda = start_lambda
        self.end_lambda = end_lambda
        self.n_steps = n_steps
        self.lin_space = np.linspace(-5, 5, n_steps)

    def step(self, step):
        if step == 0:
            return self.start_lambda
        elif step >= self.n_steps:
            return self.end_lambda
        else:
            return self.start_lambda + sigmoid(self.lin_space[step]) * self.end_lambda


