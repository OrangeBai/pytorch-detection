import torch
import torch.nn as nn
from models.blocks import *


class DNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        set_up_args = []
        set_up_kwargs = {}
        if args.batch_norm == 0:
            set_up_args.append('noBatchNorm ')
        self.layers = self.set_up(*set_up_args, **set_up_kwargs)

    def parse_layer_args(self):
        pass

    def set_up(self, *args, **kwargs):
        layers = []
        layers += [nn.Flatten()]
        layers += [LinearBlock(self.args.input_size, self.args.width, *args, **kwargs)]

        for i in range(self.args.depth - 1):
            layers += [LinearBlock(self.args.width, self.args.width, *args, **kwargs)]

        layers += [LinearBlock(self.args.width, self.args.num_cls, *args, **{'activation': None})]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
