import torch
import torch.nn as nn

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.features = make_layers(cfgs[args.net], batch_norm=True)

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(4096, args.num_cls)
        )
        self.blocks = [self.features, self.classifier]

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

        layers += [nn.Conv2d(input_channel, layer, kernel_size=(3, 3), padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(layer)]

        layers += [nn.ReLU()]
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
