import torch
import torch.nn as nn


def hook_function(self, input_var, output_var):
    output_var / self.weight.max(dim=1)[0]  # each element should be minimized
    return


if __name__ == '__main__':
    model = nn.Sequential(
        nn.Linear(in_features=100, out_features=200),
        nn.Linear(in_features=200, out_features=300),
        nn.Linear(in_features=300, out_features=400)
    )

    model.cuda()
    # x.backward()
    for layer in model.modules():
        layer.register_forward_hook(hook_function)

    x = model(torch.rand(100).cuda())

    print(1)
