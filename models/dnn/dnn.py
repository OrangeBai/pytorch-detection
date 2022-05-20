import torch
import torch.nn as nn
from models.blocks import *


class DNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layers = self.set_up()

        kwargs = {}
        if args.batch_norm == 0:
            kwargs['noBatchNorm '] = 1
        self.set_up(**kwargs)

    def parse_layer_args(self):
        pass

    def set_up(self, *args, **kwargs):
        layers = []
        layers += [nn.Flatten()]
        layers += [LinearBlock(self.args.input_size, self.args.width, *args, **kwargs)]

        for i in range(self.args.depth - 1):
            layers += [LinearBlock(self.args.input_size, self.args.width)]

        layers += [LinearBlock(self.args.width, self.args.num_cls, 'noBatchNorm', **{'activation': None})]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
