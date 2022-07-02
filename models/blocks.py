import torch.nn.functional as F

from core.pattern import *
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


class DualNet(nn.Module):
    def __init__(self, net, gamma):
        super().__init__()
        self.net = net
        self.gamma = gamma
        self.fixed_neurons = None

    def compute_float(self, x_1, x_2):
        fixed_neurons = []
        for i, module in enumerate(self.net.layers.children()):
            x_1, x_2, fix = self.compute_fix(module, x_1, x_2)
            fixed_neurons += [fix]
        self.fixed_neurons = fixed_neurons
        return x_1

    def masked_forward(self, x, fix_ratio=1, float_ratio=1):
        for i, (mask, module) in enumerate(zip(self.fixed_neurons, self.net.layers.children())):
            if type(module) == ConvBlock:
                x = module.BN(module.Conv(x))
                x = float_ratio * x * ~mask + fix_ratio * x * mask
                x = module.Act(x)
            elif type(module) == LinearBlock:
                x = module.BN(module.FC(x))
                x = float_ratio * x * ~mask + fix_ratio * x * mask
                x = module.Act(x)
            else:
                x = module(x)
        return x

    def compute_fix(self, module, x_1, x_2):
        if type(module) == ConvBlock:
            x_1 = module.Conv(x_1)
            x_2 = module.Conv(x_2)

            x_1 = self._batch_norm(module.BN, x_1)
            x_2 = self._batch_norm(module.BN, x_2)

            fixed = x_1 * x_2 > 0

            return module.Act(x_1), module.Act(x_2), fixed.detach()

        elif type(module) == LinearBlock:
            x_1 = module.FC(x_1)
            x_2 = module.FC(x_2)

            x_1 = self._batch_norm(module.BN, x_1)
            x_2 = self._batch_norm(module.BN, x_2)

            fixed = x_1 * x_2 > 0

            return module.Act(x_1), module.Act(x_2), fixed.detach()

        else:
            return module(x_1), module(x_2), None

    @staticmethod
    def _batch_norm(layer, x):
        if type(layer) in [nn.BatchNorm2d, nn.BatchNorm1d]:
            return F.batch_norm(x, layer.running_mean, layer.running_var, layer.weight, layer.bias)
        else:
            return x

    def over_fitting_forward(self, x):
        fixed_neurons = []
        for i, module in enumerate(self.net.layers.children()):
            if type(module) == ConvBlock:
                x = module.BN(module.Conv(x))
                p0 = (x < 0).sum(axis=0) > 0.99 * len(x)
                p1 = (x > 0).sum(axis=0) > 0.99 * len(x)
                p = torch.all(torch.stack([p0, p1]), dim=0).unsqueeze(dim=0)
                x = x * 1.1 * p + x * 1 * ~p
                x = module.Act(x)
            elif type(module) == LinearBlock:
                x = module.BN(module.FC(x))
                p0 = (x < 0).sum(axis=0) > 0.99 * len(x)
                p1 = (x > 0).sum(axis=0) > 0.99 * len(x)
                p = torch.all(torch.stack([p0, p1]), dim=0).unsqueeze(dim=0)
                x = x * 1.1 * p + x * 1 * ~p
                x = module.Act(x)
            else:
                x = module(x)
        return x


class FloatNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

        self.masks = None
        self.cur_block = -1

    def forward(self, x, bound=0.01, compute_type='fix', skip=4):
        self.net.eval()
        masks = []
        self.cur_block = -1
        for module in self.net.layers.children():
            x = self.block_forward(x, module)
            if type(module) in [ConvBlock, LinearBlock]:
                mask = self.compute_mask(x, bound, compute_type)
                masks += [mask]
                if self.cur_block >= skip:
                    x[~torch.tensor(mask).cuda()] = 0
        self.masks = masks
        self.net.train()
        return x

    def mask_forward(self, x, inverse=False, skip=4):
        self.net.eval()
        self.cur_block = -1
        for module in self.net.layers.children():
            x = self.block_forward(x, module)
            if type(module) in [ConvBlock, LinearBlock]:
                if self.cur_block >= skip:
                    if inverse:
                        x[torch.tensor(self.masks[self.cur_block]).cuda()] = 0
                    else:
                        x[~torch.tensor(self.masks[self.cur_block]).cuda()] = 0
        self.net.train()
        return x

    def block_forward(self, x, module):
        if type(module) == ConvBlock:
            x = self._conv_block(x, module)
            self.cur_block += 1
        elif type(module) == LinearBlock:
            x = self._linear_block(x, module)
            self.cur_block += 1
        else:
            x = module(x)
        return x

    @property
    def mask_ratio(self):
        mask_mean = []
        if self.masks is None:
            return 0
        for b_mask in self.masks:
            for l_mask in b_mask:
                mask_mean += [l_mask.mean()]
        return np.array(mask_mean).mean()

    @staticmethod
    def compute_mask(x, bound, compute_type):
        if compute_type == 'fix':
            mask = x.abs() > bound
        else:
            mask = x.abs() < bound
        return to_numpy(mask)

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
