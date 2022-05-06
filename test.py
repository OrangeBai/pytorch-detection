import torch
import torch.nn as nn


class test_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(100, 100)
        self.b = nn.ReLU()
        self.c = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 300)

        )
        self.e = nn.ReLU()
        self.d = nn.Sequential(
            nn.Linear(300, 400),
            nn.ReLU(),
            nn.Linear(400, 300)

        )
        self.x = nn.ReLU

    def forward(self, xx):
        xx = self.c(xx)
        xx = self.b(xx)
        xx = self.a(xx)
        return xx


if __name__ == '__main__':

    model = test_model().cuda()
    # x.backward()
    for name, layer in model.named_modules():
        print(layer)

    x = model(torch.rand(100).cuda())

    print(1)
