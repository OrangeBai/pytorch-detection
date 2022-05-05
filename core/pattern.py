import numpy as np
from core.utils import *
import torch
import os
from copy import deepcopy


def retrieve_input_hook(stored_values):
    """
    record input values of the module
    @param stored_values: recorder
    @return: activation hook
    """

    def hook(layer, input_var, output_var):
        input_var = input_var[0].cpu().detach()
        stored_values.append(input_var)

    return hook


def retrieve_output(stored_values):
    """
    record input values of the module
    @param stored_values: recorder
    @return: activation hook
    """

    def hook(layer, input_var, output_var):
        input_var = output_var.cpu().detach()
        stored_values.append(input_var)

    return hook


def retrieve_pattern(stored_values, Gamma):
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
        pattern = np.zeros(input_var.shape)
        num_of_pattern = len(Gamma)
        pattern[input_var < Gamma[0]] = 0
        pattern[input_var > Gamma[-1]] = num_of_pattern
        for i in range(1, num_of_pattern):
            valid = np.all([pattern > Gamma[i], pattern < Gamma[i + 1]], axis=0)
            pattern[valid] = i
        stored_values.append(pattern)

    return hook


def retrieve_fc_max(module_name, stored_value):
    pass


# def pre_activation_hook(module, )

# def add_hook(model, module_type, hook_type='pre'):
#     post_activation = {}
#
#     handles = []
#
#     layer = 0
#     for name, module in model.model.named_modules():
#         if type(module) == module_type:
#             handles += [module.register_forward_hook(set_forward_hook(layer))]
#             layer += 1
#
#     return post_activation, handles
#
#
# def add_hook(model, hook, ):
#     pass


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
        self.reset()
        for module_name, module in self.model.named_modules():
            if check_activation(module):
                self.stored_values[module_name] = []
                self.handles.append(module.register_forward_hook(
                    self.hook(self.stored_values[module_name], *self.args, **self.kwargs))
                )
        return

    def reset(self):
        for handle in self.handles:
            handle.remove()
        self.stored_values = {}

    def retrieve_res(self, name):
        try:
            return self.stored_values[name]
        except KeyError as e:
            print(e)

    def remove_handle(self, name):
        for handle in self.handles[name]:
            handle.remove()

    def calculate(self, fun, reset=True, *args, **kwargs):
        res = fun(self.stored_values, *args, **kwargs)
        if reset:
            self.set_up()
        return res


def reformat_pattern(pattern):
    r"""
    Reformat the activation pattern. forward hook to instance-wise
    @param pattern: Activation pattern recorded from forward hook function, format as:
            {layer_1: [[batch_1, neurons], [batch_2, neurons], ... [batch_n, neurons]]}
    @return: instance-wise activation pattern, format as:
            {instance_1: [{layer_1: pattern}, {layer_2: pattern}, ..., {layer_n:pattern}],
            instance_2: [{layer_1: pattern}, {layer_2: pattern}, ..., {layer_n:pattern}],
            ...,
            instance_n: [{layer_1: pattern}, {layer_2: pattern}, ..., {layer_n:pattern}]}
    """
    for name, layer in pattern.items():
        pattern[name] = np.row_stack(layer)
    num_of_instance = len(pattern[list(pattern.keys())[0]])
    for name, layer in pattern.items():
        assert len(layer) == num_of_instance

    reformatted_pattern = dict.fromkeys(range(num_of_instance), {})
    for name, layer in pattern.items():
        for idx, instance in enumerate(layer):
            reformatted_pattern[idx][name] = instance
    return reformatted_pattern


def retrieve_float_neurons(pattern):
    """
    calculate the float neurons of given pattern
    @param pattern:
    @return:
    """
    float_neurons = {}
    for name, layer in pattern.items():
        stacked_patterns = np.row_stack(pattern[name])
        float_neurons[name] = np.all(stacked_patterns, axis=0)
    return float_neurons
