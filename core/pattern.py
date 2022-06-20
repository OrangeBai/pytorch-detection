import numpy as np
from attack import *
from dataloader.base import *
from models.blocks import *
from core.utils import *


class ModelHook:
    def __init__(self, model, hook, *args, **kwargs):
        self.model = model
        self.hook = hook
        self.args = args
        self.kwargs = kwargs

        self.stored_values = {}
        self.handles = []

        self.set_up()

    def set_up(self):
        self.remove()
        for module_name, block in self.model.named_modules():
            if type(block) in [LinearBlock, ConvBlock, BottleNeck, BasicBlock]:
                self.stored_values[module_name] = {}
                self.add_block_hook(block, self.stored_values[module_name])
        return

    def add_block_hook(self, block, storage):
        for module_name, module in block.named_modules():
            if check_activation(module):
                storage[module_name] = []
                self.handles.append(module.register_forward_hook(
                    self.hook(storage[module_name], *self.args, **self.kwargs))
                )

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.stored_values = {}

    def retrieve_res(self, fun=None, reset=True, *args, **kwargs):
        if fun is not None:
            res = fun(self.stored_values, *args, **kwargs)
        else:
            res = self.stored_values
        if reset:
            self.set_up()
        return res


def set_input_hook(stored_values):
    """
    record input values of the module
    @param stored_values: recorder
    @return: activation hook
    """

    def hook(layer, input_var, output_var):
        input_var = input_var[0].cpu().detach()
        stored_values.append(input_var)

    return hook


def set_output_hook(stored_values):
    """
    record input values of the module
    @param stored_values: recorder
    @return: activation hook
    """

    def hook(layer, input_var, output_var):
        input_var = output_var.cpu().detach()
        stored_values.append(input_var)

    return hook


def set_pattern_hook(stored_values, Gamma):
    r"""
    Record the activation pattern of each neuron at this layer
    @param stored_values: recorder
    @param Gamma: A set of breakpoints, for instance,
                        if Gamma is [0],
                            the pattern of neuron is recorded as
                                0 for x_in < 0
                                1 for x_in > 0
                        if Gamma is [-1, 1]
                            the pattern of neuron is recorded as
                                0 for x_in \in (-\inf, -1)
                                1 for x_in \in (-1, 1)
                                2 for x_in \in (1, \inf)
    @return:
    """

    def hook(layer, input_var, output_var):
        input_var = input_var[0].cpu().detach()
        pattern = get_pattern(input_var, Gamma)
        stored_values.append(pattern)

    return hook


def get_pattern(input_var, Gamma):
    pattern = np.zeros(input_var.shape)
    num_of_pattern = len(Gamma)
    pattern[input_var < Gamma[0]] = 0
    pattern[input_var > Gamma[-1]] = num_of_pattern
    for i in range(1, num_of_pattern):
        valid = np.all([pattern > Gamma[i], pattern < Gamma[i + 1]], axis=0)
        pattern[valid] = i
    return pattern


def get_similarity(pattern, Gamma):
    ps = []
    for i in range(len(Gamma) + 1):
        ps_i = (pattern == i).sum(axis=0) / len(pattern)
        ps.append(ps_i)
    return np.array(ps)


def min_max_pattern(pattern, mode='min'):
    if mode == 'min':
        return pattern.min(axis=0).astype(int)
    else:
        return pattern.max(axis=0).astype(int)


def unpack(stored_values):
    unpacked = [[np.concatenate(layer)] if type(layer[0]) == np.ndarray else torch.concat(layer)
                for block in stored_values.values() for layer in block.values()]
    return unpacked


def retrieve_float_neurons(stored_values, sample_size):
    """
    calculate the float neurons of given pattern
    @param stored_values: stored value from ModelHook
    @param sample_size: size of noised samples for each input
    @return:
    """
    unpacked = unpack(stored_values)
    return [[[np.all(layer[i: i + sample_size], axis=0) for i in range(0, len(layer), sample_size)] for layer in block]
            for block in unpacked]


def retrieve_lb_ub(stored_values, grad_bound, sample_size=1):
    r"""
    Compute the upper and lower derivative bound for the pattern
    @param stored_values: recorder
    @param grad_bound: A set of gradient bounds for each activation region with length #\Gamma + 1. For instance,
                        if activation is ReLU with Gamma=[0]
                            the grad bound should be [(0,0), (1,1)]
    @param sample_size: number of samples with noise
    @return: the bound for each neuron, shape as:
                [
                    [[grad_lower_bound], [grad_upper_bound]], (instance_1),
                    [[grad_lower_bound], [grad_upper_bound]], (instance_2),
                    ...,
                    [[grad_lower_bound], [grad_upper_bound]], (instance_n)
                ]
    """

    unpacked = unpack(stored_values)

    max_lambda = np.vectorize(lambda x: grad_bound[x][1])
    min_lambda = np.vectorize(lambda x: grad_bound[x][0])

    lb = [[[min_max_pattern(layer[i: i + sample_size], 'min')
            for i in range(0, len(layer), sample_size)] for layer in block] for block in unpacked]

    ub = [[[min_max_pattern(layer[i: i + sample_size], 'max')
            for i in range(0, len(layer), sample_size)] for layer in block] for block in unpacked]
    return lb, ub


def list_all(data, storage=None):
    if type(data) == list or type(data) == tuple:
        for d in data:
            list_all(d, storage)
    elif type(data) == dict:
        for k, v in data.items():
            list_all(v, storage)
    else:
        storage.append(data)


def apd(data, storage, ed=False):
    unpacked = unpack(data)
    if len(storage) == 0:
        storage.extend(unpacked)
    else:
        for storage_block, apd_block in zip(storage, unpacked):
            for storage_layer, apd_layer in zip(storage_block, apd_block):
                storage_layer.extend(apd_layer)

    if ed:
        storage = [[to_numpy(torch.concat(layer))] for block in storage for layer in block]
    return storage

# def reformat_pattern(pattern):
#     r"""
#     Reformat the activation pattern. forward hook to instance-wise
#     @param pattern: Activation pattern recorded from forward hook function, format as:
#             {layer_1: [[batch_1, neurons], [batch_2, neurons], ... [batch_n, neurons]]}
#     @return: instance-wise activation pattern, format as:
#             {instance_1: [{layer_1: pattern}, {layer_2: pattern}, ..., {layer_n:pattern}],
#             instance_2: [{layer_1: pattern}, {layer_2: pattern}, ..., {layer_n:pattern}],
#             ...,
#             instance_n: [{layer_1: pattern}, {layer_2: pattern}, ..., {layer_n:pattern}]}
#     """
#     for name, layer in pattern.items():
#         pattern[name] = np.row_stack(layer)
#     num_of_instance = len(pattern[list(pattern.keys())[0]])
#     for name, layer in pattern.items():
#         assert len(layer) == num_of_instance
#
#     reformatted_pattern = dict.fromkeys(range(num_of_instance), {})
#     for name, layer in pattern.items():
#         for idx, instance in enumerate(layer):
#             reformatted_pattern[idx][name] = instance
#     return reformatted_pattern
