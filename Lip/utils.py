import numpy as np
from functools import reduce
from core.utils import *
from models.blocks import *
from numpy.linalg import svd


def record_blocks(model):
    """
    Record all the weights of given model
    @param model: DNN or CNN model
    @return: model weights
    """

    blocks = []
    block_types = []
    for name, module in model.model.named_modules():
        if type(module) == LinearBlock:
            blocks.append(record_linear_weights(module))
            block_types.append('linear')
        elif type(module) == ConvBlock:
            blocks.append(record_conv_weights(module))
            block_types.append('conv')
        # TODO ResBlock
        else:
            continue
    return blocks, block_types


def record_linear_weights(block):
    """
    Record Weights of Linear Blocks
    @param block: A LinearBlock module
    @return: the weights of linear layer and the batch norm
    """
    block_weights = []
    cur_weights = []
    for name, module in block.named_modules():
        if type(module) == nn.Linear:
            cur_weights.append(to_numpy(module.weight))
        elif type(module) == nn.BatchNorm1d:
            cur_weights.append(np.diag(to_numpy(module.weight) / to_numpy(torch.sqrt(module.running_var))))
        elif check_activation(module):
            cur_weights.reverse()
            block_weights.append(reduce(np.dot, cur_weights))
            cur_weights = []
    return block_weights


def record_conv_weights(block):
    block_weights = []
    for name, module in block.named_modules():
        if type(module) == nn.Conv2d:
            block_weights.append(to_numpy(module.weight))
        elif type(module) == nn.BatchNorm2d:
            block_weights.append(to_numpy(module.weight) / to_numpy(torch.sqrt(module.running_var)))
    return block_weights


def compute_jac(temp_pt, block_weights, block_types, batch_size):
    all_w = {i: [] for i in range(batch_size)}
    for pattern, weights, block_type in zip(temp_pt[:-1], block_weights[:-1], block_types[:-1]):
        for w, (k, p) in zip(weights, pattern.items()):
            for j in range(len(p)):
                all_w[j] += [w, np.diag(p[j][1])]
    [v.reverse() for v in all_w.values()]
    return [reduce(np.matmul, instance_w) for instance_w in all_w.values()]


def amplify_ratio(block_pattern, block_weight, block_type):
    if block_type == 'linear':
        return linear_amplify(block_pattern, block_weight)
    elif block_type == 'conv':
        pass


def linear_amplify(pattern, weight):
    r = []
    for w, (k, p) in zip(weight, pattern.items()):
        for i in range(len(p)):
            fixed_matrix = np.matmul(np.diag(p[i]), w)
            float_matrix = np.matmul(np.diag(1 - p[i]), w)
            r += [1 + svd(float_matrix)[1][0] / svd(fixed_matrix)[1][0]]
    return np.array(r)
