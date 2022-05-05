import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layers = self.set_up()

        self.set_up()

    def set_up(self):
        layers = []
        layers += [nn.Flatten()]
        layers += [nn.Linear(self.args.input_size, self.args.width)]
        layers += [nn.BatchNorm1d(self.args.width)]
        layers += [nn.ReLU()]

        for i in range(self.args.depth):
            layers += [nn.Linear(self.args.width, self.args.width)]
            layers += [nn.BatchNorm1d(self.args.width)]
            layers += [nn.ReLU()]

        layers += [nn.Linear(self.args.width, self.args.num_cls)]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
