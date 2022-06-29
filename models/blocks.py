import torch.nn as nn
import torch.nn.functional as F
from core.utils import *

def set_activation(activation):
    if activation is None:
        return nn.Identity()
    elif activation.lower() == 'relu':
        return nn.ReLU(inplace=False)
    elif activation.lower() == 'prelu':
        return nn.PReLU()
    elif activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(0.1)


def set_bn(batch_norm, dim, channel):
    if not batch_norm:
        return nn.Identity()
    else:
        if dim == 1:
            return nn.BatchNorm1d(channel)
        else:
            return nn.BatchNorm2d(channel)


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()

        self.FC = nn.Linear(in_channels, out_channels)
        self.BN = set_bn(kwargs['batch_norm'], dim=1, channel=out_channels)
        self.Act = set_activation(kwargs['activation'])

    def forward(self, x):
        x = self.FC(x)
        x = self.BN(x)
        x = self.Act(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, *args, **kwargs):
        super().__init__()
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.BN = set_bn(kwargs['batch_norm'], 2, out_channels)
        self.Act = set_activation(kwargs['activation'])

    def forward(self, x):
        x = self.Conv(x)
        x = self.BN(x)
        x = self.Act(x)
        return x


class FloatConv(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x, mask):
        x = self.conv(x)
        x[mask] = 0
        return x


class FloatFC(nn.Module):
    def __init__(self, fc):
        super().__init__()
        self.fc = fc

    def forward(self, x, mask):
        x = self.fc(x)
        x[mask] = 0
        return x


class FloatNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def first_forward(self, x, bound, compute_type='fix'):
        masks = []
        for module in self.net.layers.children():
            if type(module) == ConvBlock:
                x = self._conv_block(x, module)
                x, mask = self.compute_mask(x, bound, compute_type)
                masks += [mask]
            elif type(module) == LinearBlock:
                x = self._linear_block(x, module)
                x, mask = self.compute_mask(x, bound, compute_type)
                masks += [mask]
            else:
                x = module(x)

        return x, masks

    def forward(self, x, masks, inverse=False):
        mask_mean = 0
        for module, mask in zip(self.net.layers.children(), masks):
            if type(module) == ConvBlock:
                x = self._conv_block(x, module)
                if inverse:
                    x[mask] = 0
                else:
                    x[~mask] = 0
                mask_mean += mask.mean()
            elif type(module) == LinearBlock:
                x = self._linear_block(x, module)
                if inverse:
                    x[mask] = 0
                else:
                    x[~mask] = 0
                mask_mean += mask.mean()
            else:
                x = module(x)

        return x, mask_mean

    @staticmethod
    def compute_mask(x, bound, compute_type):
        if compute_type == 'fix':
            mask = x.abs() > bound
        else:
            mask = x.abs() < bound
        x[~mask] = 0
        return x, to_numpy(mask)

    def _conv_block(self, x, module):
        x = self._conv2d(x, module.Conv.weight, module.Conv.bias, module.Conv.stride, padding=module.Conv.padding)
        x = module.BN(x)
        x = module.Act(x)
        return x

    def _linear_block(self, x, module):
        x = self._linear(x, module.FC.weight, module.FC.bias)
        x = module.BN(x)
        x = module.Act(x)
        return x

    @staticmethod
    def _conv2d(x, w, b, stride=1, padding=0):
        return F.conv2d(x, w, bias=b, stride=stride, padding=padding)

    @staticmethod
    def _linear(x, w, b):
        return F.linear(x, w, b)


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, act='ReLU', *args, **kwargs):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            set_activation(act),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        self.act2 = set_activation(act)

    def forward(self, x):
        return self.act2(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, act='ReLU', *args, **kwargs):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            set_activation(act),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            set_activation(act),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        self.act2 = set_activation(act)

    def forward(self, x):
        return self.act2(self.residual_function(x) + self.shortcut(x))
