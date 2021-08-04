import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


class JointAdamOptVQVAE:
    def __init__(self, model, lr_ed, lr_q):
        self.optimizer_ed = optim.Adam(model.get_encoder_decoder_params(), lr=lr_ed)
        self.optimizer_q = optim.Adam(model.get_quantizer_params(), lr=lr_q)

    def step(self):
        self.optimizer_ed.step()
        self.optimizer_q.step()

    def zero_grad(self):
        self.optimizer_ed.zero_grad()
        self.optimizer_q.zero_grad()


class JointMultiStepLrVQVAE:
    def __init__(self, optimizer, milestones, gamma):
        self.lr_scheduler_ed = MultiStepLR(optimizer.optimizer_ed, milestones=milestones, gamma=gamma)
        self.lr_scheduler_q = MultiStepLR(optimizer.optimizer_q, milestones=milestones, gamma=gamma)

    def step(self):
        self.lr_scheduler_ed.step()
        self.lr_scheduler_q.step()


class AdamOptWithMultiStepLrVQVAE:
    def __init__(self, type, model, lr, lr_q, milestones, gamma):
        possible_types = ['joint', 'vocab_sep']
        if type not in possible_types:
            raise ValueError('Type {} is non-existent.'.format(type))
        self.type = type
        if self.type == possible_types[0]:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
            self.lr_scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        elif self.type == possible_types[1]:
            self.optimizer = JointAdamOptVQVAE(model, lr_ed=lr, lr_q=lr_q)
            self.lr_scheduler = JointMultiStepLrVQVAE(self.optimizer, milestones=milestones, gamma=gamma)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step(self):
        self.lr_scheduler.step()





