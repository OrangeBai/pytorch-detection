import torch
import torch.nn as nn

class test_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(100, 100)
        self.b = nn.ReLU()
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
        xx = self.a(xx)
        # xx = self.b(xx)
        # xx = self.c(xx)
        return xx


if __name__ == '__main__':
    torch.set_grad_enabled(True)

    model = test_model().cuda()
    # x.backward()
    for name, layer in model.named_modules():
        print(layer)
        layer.require_grad=True
    i = torch.rand(100, 100, requires_grad=True).cuda()

    x = model(i.cuda())
    x_norm = x.norm()
    grad = torch.autograd.grad(x_norm, i)

    print(1)
