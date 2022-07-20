import time

from core.pattern import *


def compute_jac_cnn(net, img, activation='ReLU'):
    t1 = time.time()
    batch_x = []
    _, c, h, w = img.shape
    for i in range(c):
        for j in range(h):
            for k in range(w):
                basis = torch.zeros_like(img[0])
                basis[i, j, k] = 1
                batch_x += [basis]
    batch_x = torch.stack(batch_x)
    net.eval()
    x = img.detach()
    for module in net.children():
        if type(module) == LinearBlock:
            x = module.BN(module.FC(x))
            pattern = get_pattern(x, Gamma=set_gamma(activation))
            _, ub = layer_lb_ub(pattern, grad_bound=[(0, 0), (1, 1)])

            batch_x = fc(batch_x, module.FC)
            batch_x = batch_norm(batch_x, module.BN)
            if type(module.Act) != nn.Identity:
                batch_x = act_pattern(batch_x, ub)
        elif type(module) == ConvBlock:
            x = module.BN(module.Conv(x))
            pattern = get_pattern(x, Gamma=set_gamma(activation))
            _, ub = layer_lb_ub(pattern, grad_bound=[(0, 0), (1, 1)])

            batch_x = conv(batch_x, module.Conv)
            batch_x = batch_norm(batch_x, module.BN)
            if type(module.Act) != nn.Identity:
                batch_x = act_pattern(batch_x, ub)
        elif type(module) == nn.MaxPool2d:
            x2, indices = F.max_pool2d(x, module.kernel_size, module.stride, return_indices=True)

            flattened_tensor = x.flatten(start_dim=2)
            output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)

            indices = torch.concat([indices] * len(batch_x))
            flattened_tensor = batch_x.flatten(start_dim=2)
            batch_x = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)

            x = x2
        else:
            x = module(x)
            batch_x = module(batch_x)

    t2 = time.time()
    print(t2 - t1)
    img.requires_grad = True
    # y = net(img)[0]
    # g = []
    # for i in range(len(y)):
    #     g += [torch.autograd.grad(y[i], img, retain_graph=True)[0]]

    # g = torch.concat(g)
    j = torch.autograd.functional.jacobian(net, images)
    print(time.time() - t2)
    permuted = j.squeeze().permute(1, 2, 3, 0)
    diff = permuted - batch_x.view(3, 32, 32, -1)
    return batch_x


def fc(x, fc_layer):
    return F.linear(x, fc_layer.weight)


def act_pattern(x, ub):
    ub = torch.tensor(ub, dtype=torch.float).unsqueeze(dim=0)
    return x * ub


def batch_norm(x, bn_layer):
    if type(bn_layer) == nn.Identity:
        return x
    else:
        return F.batch_norm(x, running_mean=torch.zeros_like(bn_layer.running_mean),
                            running_var=bn_layer.running_var, weight=bn_layer.weight)


def conv(x, conv_layer):
    return F.conv2d(x, conv_layer.weight.data, bias=None, stride=conv_layer.stride, padding=conv_layer.padding)


if __name__ == '__main__':
    kwargs = {'activation': 'ReLU', 'batch_norm': 1}
    layers = [
        ConvBlock(3, 32, 3, batch_norm=1, padding=1, stride=2,  activation='ReLU'),
        ConvBlock(32, 32, 3, padding=1, stride=1, activation='ReLU', batch_norm=1),
    #
              # ConvBlock(16, 16, batch_norm=True, padding=1, stride=2, activation='ReLU'),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Flatten(),
              LinearBlock(2048, 10, batch_norm=0, activation=None)
              ]
    model = nn.Sequential(*layers)
    # model = nn.Sequential(
    #     ConvBlock(3, 32, 3, padding=1, **kwargs),
    #     ConvBlock(32, 32, 3, padding=1, stride=2, **kwargs),
    #     # nn.MaxPool2d(2, 2),
    #     ConvBlock(32, 64, 3, padding=1, **kwargs),
    #     ConvBlock(64, 64, 3, padding=1, stride=2, **kwargs),
    #     # nn.MaxPool2d(2, 2),
    #     nn.Flatten(),
    #     LinearBlock(64 * 8 * 8, 512, **kwargs),
    #
    #     LinearBlock(512, 512, **kwargs),
    #     LinearBlock(512, 10, batch_norm=1, activation=None),
    # )

    # net = nn.Sequential(*layers)
    images = torch.randn((1, 3, 32, 32))
    # images = torch.randn(1, 1, 1, 4096)
    compute_jac_cnn(model, images)

