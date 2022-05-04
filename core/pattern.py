import numpy as np
from core.utils import *
import torch
import os


def retrieve_input_hook(module_name, stored_values):
    def hook(layer, input_var, output_var):
        input_var = input_var[0].cpu().detach()
        stored_values[module_name].append(input_var)

    return hook


def retrieve_fc_max(module_name, stored_value):
    pass


def retrieve_output(module_name, stored_values):
    def hook(layer, input_var, output_var):
        input_var = output_var.cpu().detach()
        stored_values[module_name].append(input_var)

    return hook


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
    def __init__(self, model):
        self.model = model
        self.stored_values = {}
        self.hooks = []
        self.handles = {}

    def add_hook(self, hook_name, hook, *args, **kwargs):
        cur_stored_val = dict()
        cur_handles = []
        for module_name, module in self.model.named_modules():
            if check_activation(module):
                cur_stored_val[module_name] = []
                cur_handles.append(module.register_forward_hook(hook(module_name, cur_stored_val)))
        self.stored_values[hook_name] = cur_stored_val
        self.handles = cur_handles
        return

    def retrieve_res(self, name):
        try:
            return self.stored_values[name]
        except KeyError as e:
            print(e)

    def remove_handle(self, name):
        for handle in self.handles[name]:
            handle.remove()

    def reset(self):
        self.stored_values = []
        for handle in self.handles:
            handle.remove()

    def pop(self, name):
        self.stored_values.pop(name)

nn.Conv2d