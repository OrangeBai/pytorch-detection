import torch
import torch.nn as nn


def set_activation(name):
    if name is None:
        return nn.Identity()
    elif name == 'relu':
        return nn.ReLU(inplace=True)


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()

        self.layers = [nn.Linear(in_channels, out_channels)]
        if 'noBatchNorm' in args:
            self.layers += [nn.BatchNorm1d(out_channels)]
        if 'activation' in kwargs.keys():
            self.layers += [set_activation(kwargs['activation'])]
        else:
            self.layers += [nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1, *args, **kwargs):
        super().__init__()
        self.layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
        if 'noBatchNorm' not in kwargs.keys():
            self.layers += [nn.BatchNorm2d(out_channels)]
        if 'activation' in kwargs.keys():
            self.layers += [set_activation(kwargs['activation'])]
        else:
            self.layers += [nn.ReLU(inplace=True)]

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
