import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_cls = args.num_cls
        self.layers = []

        self.set_up(make_layers(cfg, batch_norm=True))