from core.utils import *
from models.blocks import *


def record_blocks(model):
    blocks = []

    for name, module in model.model.named_modules():
        if type(module) == LinearBlock:
            blocks.append(record_linear_weights(module))
        elif type(module) == ConvBlock:
            blocks.append(record_conv_weights(module))
        # elif type(module) == torch.nn.BatchNorm1d:
        #     bn_factor = to_numpy(module.weight) / to_numpy(torch.sqrt(module.running_var))
        #     weights.append(bn_factor)
        # elif check_activation(module):
        #     cur_block.append()


def record_linear_weights(block):
    block_weights = []
    for name, module in block.named_modules():
        if type(module) == nn.Linear:
            block_weights.append(to_numpy(module.weight))
        elif type(module) == nn.BatchNorm1d:
            block_weights.append(to_numpy(module.weight) / to_numpy(torch.sqrt(module.running_var)))
    return block_weights


def record_conv_weights(block):
    block_weights = []
    for name, module in block.named_modules():
        if type(module) == nn.Conv2d:
            block_weights.append(to_numpy(module.weight))
        elif type(module) == nn.BatchNorm2d:
            block_weights.append(to_numpy(module.weight) / to_numpy(torch.sqrt(module.running_var)))
    return block_weights
