import torch
from torch.nn import functional as F

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


class DualNet(nn.Module):
    def __init__(self, net, args):
        super().__init__()
        self.net = net
        self.eta_dn = args.eta_dn
        self.dn_rate = args.dn_rate
        self.gamma = set_gamma(args.activation)
        self.num_layers = args.eta_layers
        self.block_len = self.count_block_len
        self.fixed_neurons = []

    @property
    def count_block_len(self):
        counter = 0
        for module in self.net.layers.children():
            if type(module) in [ConvBlock, LinearBlock]:
                counter += 1
        return counter

    def forward(self, x_1, x_2, eta_fixed=0, eta_float=0):
        fixed_neurons = []
        df = torch.tensor(1, dtype=torch.float).cuda()
        counter = 0
        for i, module in enumerate(self.net.layers.children()):
            x_1 = self.compute_pre_act(module, x_1)
            x_2 = self.compute_pre_act(module, x_2)
            if self.check_block(module):
                if counter >= self.block_len - self.num_layers:
                    fix = self.compute_fix(x_1, x_2)
                    df += ((x_1 - x_2) * ~fix).abs().mean()
                    x_1 = self.x_mask(x_1, eta_fixed, fix) + self.x_mask(x_1, eta_float, ~fix)
                    x_2 = self.x_mask(x_2, eta_fixed, fix) + self.x_mask(x_2, eta_float, ~fix)
                    fixed_neurons += [fix]
                    df += ((x_1 - x_2) * fix).view(len(x_1), -1).norm(p=2, dim=-1).mean()
                x_1 = module.Act(x_1)
                x_2 = module.Act(x_2)

                counter += 1
        self.fixed_neurons = fixed_neurons
        return x_1, x_2, df

    def masked_forward(self, x):
        for i, (fix, module) in enumerate(zip(self.fixed_neurons, self.net.layers.children())):
            x = self.compute_pre_act(module, x)
            if type(module) in [ConvBlock, LinearBlock]:
                x = self.x_mask(x, -1, ~fix) + self.x_mask(x, 0, fix)
                x = module.Act(x)
        return x

    @staticmethod
    def check_block(module):
        if type(module) in [ConvBlock, LinearBlock]:
            return 1
        else:
            return 0

    def dn_forward(self, x):
        for i, module in enumerate(self.net.layers.children()):
            x = self.compute_pre_act(module, x)
            if type(module) in [ConvBlock, LinearBlock]:
                p0 = (x < 0).sum(axis=0) > self.dn_rate * len(x)
                p1 = (x > 0).sum(axis=0) > self.dn_rate * len(x)
                p_same = torch.all(torch.stack([p0, p1]), dim=0).unsqueeze(dim=0)
                x = self.x_mask(x, self.eta_dn, p_same) + x * ~p_same
                x = module.Act(x)
        return x

    @property
    def mask_ratio(self):
        mask_mean = []
        if len(self.fixed_neurons) == 0:
            return 0
        for b_mask in self.fixed_neurons:
            if b_mask is not None:
                mask_mean += [to_numpy(b_mask).mean()]
            return np.array(mask_mean).mean()

    @staticmethod
    def x_mask(x, ratio, mask):
        return x * (1 + ratio) * mask - x.detach() * mask.detach() * ratio

    @staticmethod
    def compute_pre_act(module, x):
        if type(module) == ConvBlock:
            return module.BN(module.Conv(x))
        elif type(module) == LinearBlock:
            return module.BN(module.FC(x))
        else:
            return module(x)

    @staticmethod
    def compute_fix(x_1, x_2):
        fixed = x_1 * x_2 > 0
        return fixed.detach()

    @staticmethod
    def _batch_norm(layer, x):
        if type(layer) in [nn.BatchNorm2d, nn.BatchNorm1d]:
            return F.batch_norm(x, layer.running_mean, layer.running_var, layer.weight, layer.bias)
        else:
            return x


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
