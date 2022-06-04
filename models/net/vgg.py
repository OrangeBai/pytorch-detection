import torch
import torch.nn as nn
from models.blocks import *

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.num_cls = args.num_cls
        self.layers = []

        if args.net.lower() == 'vgg11':
            cfg = cfgs['vgg11']
        elif args.net.lower() == 'vgg13':
            cfg = cfgs['vgg13']
        elif args.net.lower() == 'vgg16':
            cfg = cfgs['vgg16']
        elif args.net.lower() == 'vgg19':
            cfg = cfgs['vgg19']
        else:
            raise NameError("No network named {}".format(args.net))
        self.set_up(make_layers(cfg, batch_norm=True), args.model_type)

    def set_up(self, features, model_type):
        setattr(self, 'features', features)
        if model_type == 'mini':
            setattr(self, 'avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
        else:
            setattr(self, 'avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
        setattr(self, 'classifier', nn.Sequential(
            LinearBlock(512, 4096),
            # nn.Dropout(),
            LinearBlock(4096, 4096),
            nn.ReLU(),
            # nn.Dropout(),
            LinearBlock(4096, self.num_cls, **{'activation': None})
        ))
        self.layers = [self.features, self.classifier]

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for layer in cfg:
        if layer == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [ConvBlock(input_channel, layer, kernel_size=(3, 3), padding=1)]
        input_channel = layer

    return nn.Sequential(*layers)


class VGG11(VGG):
    def __init__(self, args):
        super().__init__(args)


class VGG13(VGG):
    def __init__(self, args):
        super().__init__(args)


class VGG16(VGG):
    def __init__(self, args):
        super().__init__(args)


class VGG19(VGG):
    def __init__(self, args):
        super().__init__(args)
