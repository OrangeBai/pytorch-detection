import os
import numpy as np
import torch
from settings.test_setting import *


def plot(args):
    argv = ['--dir', '', '--exp_id', 'st', '--net', 'vgg16', '--dataset', 'cifar10',
            '--batch_norm', '1', '--batch_size', '1',
            '--test_name', 'ap_lip']
    exp_args = set_up_testing()
    data_cg = torch.load(os.path.join(args.exp_path, 'data_cg'))
    data_points = torch.load(os.path.join(args.exp_path, 'data_points'))
    dd = np.argmax(np.any(data_cg[:, :, 1] != 0, axis=1), axis=1)

    rt_steps = []
    rt_distance = []
    for data in data_points:
        for layer in data:
            returned_idx = np.where(layer[1] != 0)
            if len(returned_idx) != 0:
                rt_step = (layer[1] - layer[0])[returned_idx].tolist()
                rt_dis = layer[1][returned_idx].tolist()
                rt_steps.extend(rt_step)
                rt_distance.extend(rt_dis)


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # h = ax.hist2d(rt_distance, np.array(rt_steps) / np.array(rt_distance), bins=100, norm=LogNorm())
    h=ax.hist2d(rt_distance, rt_steps, bins=100)
    fig.colorbar(h[3], ax=ax)
    plt.show()
    print(1)
