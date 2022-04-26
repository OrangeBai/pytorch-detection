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


class test_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(2, 5)
        self.act = nn.ReLU(inplace=True)
        self.out = nn.Linear(5, 4)

    def forward(self, x):
        out = self.dense(x)
        out = self.act(out)
        out = self.out(out)
        return out


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
y = np.array(y) - 1
a = get_close_points(x, 400)
for i in range(4):
    plt_points(plt, a[i], 5, 0.2, i)
a = get_close_points(x, 30)
for i in range(4):
    plt_points(plt, a[i], 200, 1, i, fill=False)
plt.savefig('random.png')

test_x = x[::4]
test_y = y[::4]
index = [np.where(x == d) for d in test_x]
index = np.array(index)[:, 0, 0].tolist()
train_x = x[index]
train_y = y[index]

train_dataset = TensorDataset(torch.Tensor(train_x), torch.tensor(train_y, dtype=torch.long))
test_dataset = TensorDataset(torch.Tensor(test_x), torch.tensor(test_y, dtype=torch.long))

train_dataloader = DataLoader(train_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

criterion = nn.CrossEntropyLoss()
train_loss = 0
train_acc = 0
train_n = 0
adv_images = [[], [], [], []]
acc = np.zeros((10, 100))
test_acc = np.zeros((10, 100))
for num in range(10):
    model = test_model()
    model.cuda()

    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    for i in range(100):
        train_loss = 0
        train_acc = 0
        train_n = 0
        for _, (x, y) in enumerate(train_dataloader):
            x = x.cuda()
            y = y.cuda()
            # x = fgsm(model, x, y)
            out = model(x)
            loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (out.max(1)[1] == y).sum().item()
            train_n += y.size(0)
        # if i % 5 == 0 and i != 0:
        #     for j in range(4):
        #         test_model = deepcopy(model)
        #         if i == 5:
        #             adv_images[j] = fgsm(test_model, torch.tensor(a[j], dtype=torch.float),
        #                                  torch.tensor([j] * a[j].shape[0], dtype=torch.long))
        #             adv_images[j] = adv_images[j].cpu().detach().numpy()
        #             plt_lines(plt, adv_images[j], a[j])
        #             plt_points(plt, adv_images[j], 5, 1, j)
        #         else:
        #             adv_images2 = deepcopy(adv_images)
        #             adv_images[j] = fgsm(test_model, torch.tensor(a[j], dtype=torch.float),
        #                                  torch.tensor([j] * a[j].shape[0], dtype=torch.long))
        #             adv_images[j] = adv_images[j].cpu().detach().numpy()
        #             plt_lines(plt, adv_images[j], adv_images2[j])
        #             plt_points(plt, adv_images[j], 5, 1, j)
        test_acc[num, i] = test(test_dataloader, model, 'pgd')
        acc[num, i] = (train_acc / train_n)
        print(train_acc / train_n)
# plt.savefig('fgsm.png')
np.save('natural.npy', acc)
np.save('natural_test.npy', test_acc)
print(1)
