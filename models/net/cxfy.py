import models.net.cxfy
from models.blocks import *
import math
cfgs = {
    'c4f2': [32, 32, 64, 64],
}


class CXFY(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_cls = args.num_cls
        self.model = self.set_up()

    def set_up(self):
        model = getattr(models.net.cxfy, '_'.join([self.args.net.lower(), self.args.dataset.lower(), self.args.shape]))
        return model()

    def forward(self, x):
        return self.model(x)

def cxfy42_mnist_large():
    model = nn.Sequential(
        ConvBlock(1, 32, 3, padding=1),
        ConvBlock(32, 32, 3, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        ConvBlock(32, 64, 3, padding=1),
        ConvBlock(64, 64, 3, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def cxfy42_cifar10_large():
    model = nn.Sequential(
        ConvBlock(3, 32, 3, padding=1),
        ConvBlock(32, 32, 3, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        ConvBlock(32, 64, 3, padding=1),
        ConvBlock(64, 64, 3, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        LinearBlock(64 * 8 * 8, 512),
        nn.ReLU(),
        LinearBlock(512, 512),
        nn.ReLU(),
        LinearBlock(512, 10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model
