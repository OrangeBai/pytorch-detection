import torch
import torch.nn as nn

if __name__ == '__main__':
    layers = [nn.Conv2d(3, 16, (3, 3), (1, 1), padding=1, bias=False),
              # nn.BatchNorm2d(16),
              nn.ReLU(),
              nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1, bias=False),
              # nn.BatchNorm2d(16),
              nn.ReLU(),
              nn.Conv2d(16,16,(3,3), (1,1), padding=1, bias=False),
              # nn.BatchNorm2d(16),
              nn.ReLU(),
              nn.Flatten()
              ]
    net = nn.Sequential(*layers)
    x = torch.randn((1, 3, 16, 16))
    x.requires_grad = True
    y = net(x)
    a = torch.autograd.functional.jacobian(net, x).squeeze()
    b = a[85, 0].numpy()
    c = a[87, 0].numpy()
    d = net[0].weight
    print(1)
    res = torch.zeros(4096, 3, 16, 16)
    x2 = torch.zeros_like(x)
    x2[0, 0, 8, 8] = 1
    e = net(x2)
    print((e[0] - a[:, 0, 8,8]).max())
    print(1)

