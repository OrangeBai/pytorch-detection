from core.pattern import *
from core.utils import *


def find_dead_neuron(storage, Gamma):
    block_same = []
    for block in storage:
        layer_same = []
        for layer in block:
            pattern = get_pattern(layer, Gamma)
            neuron_ps = get_similarity(pattern, Gamma) * len(layer)
            if len(layer.shape) > 1:
                sum_axis = tuple(range(1, len(layer.shape) - 1))
                layer_ps = neuron_ps.max(axis=0).mean(axis=sum_axis)
            else:
                layer_ps = neuron_ps
            layer_same.append(layer_ps)
        block_same.append(layer_same)
    return block_same


def prune_block(m, cur_block_ps, pre_ps, conv_ratio=0.95, linear_ratio=0.98):
    if type(m) == ConvBlock:
        activate_filters = cur_block_ps[0] <= conv_ratio
        m.Conv.weight.data = m.Conv.weight.data[activate_filters]
        m.Conv.bias.data = m.Conv.bias.data[activate_filters]

        if len(pre_ps) != 0:
            pre_activate_filters = pre_ps[0] <= conv_ratio
            m.Conv.weight.data = m.Conv.weight.data[:, pre_activate_filters]

        m.BN.weight.data = m.BN.weight.data[activate_filters]
        m.BN.bias.data = m.BN.bias.data[activate_filters]
        m.BN.running_mean.data = m.BN.running_mean.data[activate_filters]
        m.BN.running_var.data = m.BN.running_var.data[activate_filters]
        c = activate_filters.sum()
    elif type(m) == LinearBlock:
        if len(cur_block_ps) == 0:
            c = None
        else:
            activate_filters = cur_block_ps[0] <= linear_ratio
            m.FC.weight.data = m.FC.weight.data[activate_filters]
            m.FC.bias.data = m.FC.bias.data[activate_filters]
            m.BN.weight.data = m.BN.weight.data[activate_filters]
            m.BN.bias.data = m.BN.bias.data[activate_filters]
            m.BN.running_mean = m.BN.running_mean[activate_filters]
            m.BN.running_var = m.BN.running_var[activate_filters]
            c = activate_filters.sum()

        pre_activate_filters = np.where(pre_ps[0] <= linear_ratio)
        m.FC.weight.data = m.FC.weight.data.T[pre_activate_filters].T

    else:
        raise TypeError

    return m, c


def compute_mean(net_mean_all, net_mean):
    if net_mean_all is None:
        return net_mean
    else:
        for i, (block_all, block) in enumerate(zip(net_mean_all, net_mean)):
            for j, (layer_all, layer) in enumerate(zip(block_all, block)):
                net_mean_all[i][j] = (layer_all + layer)
    return net_mean_all
