
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
from attacks import *
from torchvision import transforms
from swiss_roll.attack import *
import matplotlib.pyplot as plt
from copy import deepcopy

x = []
y = []
with open('data.txt') as f:
    for line in f:
        str_data = line.strip().split('  ')
        x.append([float(data) for data in str_data])
x = np.array(x)
with open('label.txt') as f:
    for line in f:
        str_data = line.strip()
        y.append(float(str_data))

x = (x - x.min()) / (x.max() - x.min())
xx = []
for i in range(4):
    xx += [x[i*400:i*400+400]]

mean = np.array(xx).mean(axis=1)

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          # 'xtick.labelsize': med,
          # 'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-dark')


y = np.array(y) - 1
a = get_close_points(x, 400)
fig, ax = plt.subplots()
for i in range(4):
    pt = []
    for j in xx[i]:
        if np.abs((j - mean[i])).max() < 0.18:
            pt += [j]
    plt_points(ax, np.array(pt), 8, 0.5, i)


ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

fig.savefig(r'F:\PHD\Computer science\_Paper\EAI\submission\AAAI-22\LaTeX\sections\figures\Illuss.png', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.2)
