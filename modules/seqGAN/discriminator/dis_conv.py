import os
import torch
from torch import nn
import torch.nn.functional as F
from modules.common_blocks import ResidualStack, DownSampleX2, ChangeChannels


class Discriminator28(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 bias=True,
                 use_bn=True):
        super(Discriminator28, self).__init__()

        self.blocks = nn.Sequential(
            ChangeChannels(in_channels=in_channels, out_channels=hidden_dim, bias=bias, use_bn=use_bn),
            DownSampleX2(in_channels=hidden_dim, out_channels=hidden_dim, bias=bias, use_bn=use_bn),
            ResidualStack(in_channels=hidden_dim, out_channels=hidden_dim, num_residual_layers=1,
                          bias=bias, use_bn=use_bn, final_relu=True),
            DownSampleX2(in_channels=hidden_dim, out_channels=hidden_dim, bias=bias, use_bn=use_bn),
            ResidualStack(in_channels=hidden_dim, out_channels=hidden_dim, num_residual_layers=1,
                          bias=bias, use_bn=use_bn, final_relu=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=0, bias=bias),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=0, bias=bias),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=1)
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.classifier(x.squeeze())
        return torch.sigmoid(x.squeeze())

    def save_model(self, root_path, model_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        path = os.path.join(root_path, model_name + ".pth")
        torch.save(self.state_dict(), path)

    def load_model(self, root_path, model_name, map_location=torch.device('cpu')):
        path = os.path.join(root_path, model_name + ".pth")
        self.load_state_dict(torch.load(path, map_location=map_location))

