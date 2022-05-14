import torch
import torch.nn as nn
import numpy as np
from core.funcs import *
import torch.nn.functional as F


class test_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(100, 100)
        self.b = nn.ReLU()
        self.c = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.d = nn.Flatten()
        self.e = nn.Linear(in_features=32 * 8 * 8, out_features=10)
        # self.c = nn.Sequential(
        #     nn.Linear(100, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 300)
        #
        # )
        # self.e = nn.ReLU()
        # self.d = nn.Sequential(
        #     nn.Linear(300, 400),
        #     nn.ReLU(),
        #     nn.Linear(400, 300)
        #
        # )
        # self.x = nn.ReLU()

    def forward(self, xx):
        xx = self.c(xx)
        xx = self.d(xx)
        xx = self.e(xx)
        # xx = self.d(xx)
        # xx = self.b(xx)
        # xx = self.c(xx)
        return xx


if __name__ == '__main__':
    model = test_model().cuda()

    x = torch.randn((1, 3, 8, 8), requires_grad=True).cuda()
    y = model(x)

    xx = x.cpu().detach().numpy()
    ww = model.c.weight.detach().cpu().numpy()
    www = ff_dis(ww)
    xx_d = torch.Tensor(ff_dis(xx, padding=1)).cuda()

    asdf = F.conv2d(x, torch.Tensor(www).cuda(), padding=1)

    c = np.linalg.svd(torch.autograd.functional.jacobian(model, x).squeeze().reshape((10, -1)).cpu().detach().numpy())
    print(1)
    # df = torch.Tensor(ff_dis(x.cpu(), 1)).cuda()
    # c = np.fft.fft2(ww, axes=[-2, -1])
    # torch.autograd.functional.jacobian(model.model, x)
    #     ff_in = torch.fft.fft2(torch.randn((1, 32, 32, 32))).cpu().detach().numpy()
    # F.conv2d(ff_in, w)
    # x.backward()
    # for name, layer in model.named_modules():
    #     print(layer)
    #     layer.require_grad=True
    # i = torch.rand(100, 100, requires_grad=True).cuda()

    # x = model(i.cuda())
    # x_norm = x.norm()
    # grad = torch.autograd.grad(x_norm, i)

    print(1)
