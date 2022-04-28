import torch.nn as nn
from core.utils import *


class Attack(object):
    def __init__(self, name, model, mean=None, std=None):
        if std is None:
            std = [1, 1, 1]
        if mean is None:
            mean = [0, 0, 0]
        self.name = name
        self.norm_layer = Normalize(mean=mean, std=std)
        self.model = nn.Sequential(self.norm_layer, model).cuda()
        self.mean = torch.tensor(mean).view(len(mean), 1, 1).cuda()
        self.std = torch.tensor(std).view(len(mean), 1, 1).cuda()

        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)

    def _reverse_norm(self, x):
        return x * self.std + self.mean

    def _norm(self, x):
        return (x - self.mean) / self.std

    def attack(self, *args):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std
