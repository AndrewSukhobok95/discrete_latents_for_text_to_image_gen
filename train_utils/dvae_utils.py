import torch
import numpy as np


def KLD_loss(z_dist):
    eps = 1e-20
    b, lat_dim, h, w = z_dist.size()
    z_dist_flatten = z_dist.view(b, -1)
    log_ratio = torch.log(z_dist_flatten * lat_dim + eps)
    KLD = torch.sum(z_dist_flatten * log_ratio, dim=-1).mean()
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


