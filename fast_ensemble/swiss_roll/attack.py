from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


def fgsm(model, x, y, eps=0.02):
    loss = nn.CrossEntropyLoss()
    x = x.cuda()
    y = y.cuda()

    x.requires_grad = True
    outputs = model(x)
    cost = loss(outputs, y)

    grad = torch.autograd.grad(cost, x,
                               retain_graph=False, create_graph=False)[0]

    adv_images = x + eps * grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    return adv_images


def pgd(model, images, labels, eps=0.02, alpha=0.01):
    r"""
    Overridden.
    """
    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    loss_fn = nn.CrossEntropyLoss()

    max_loss = torch.zeros(labels.shape[0]).cuda()
    max_delta = torch.zeros_like(images).cuda()

    for zz in range(2):
        delta = torch.zeros_like(images).cuda()
        delta.uniform_(-eps, eps)
        delta.data = torch.clamp(delta, min=-eps, max=eps)
        delta.requires_grad = True
        for i in range(10):
            outputs = model(images + delta)
            loss = loss_fn(outputs, labels)
            loss.backward()
            grad = delta.grad.detach()
            d = delta
            g = grad
            d = torch.clamp(d + alpha * torch.sign(g), -eps, eps)
            d = clamp(d, 0 - images, 1 - images)
            delta.data = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(images + delta), labels, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return images + max_delta


def get_close_points(x, n):
    x1 = x[:400]
    x2 = x[400:800]
    x3 = x[800:1200]
    x4 = x[1200:]
    xx = [x1, x2, x3, x4]
    points = [[], [], [], []]
    for i in range(4):
        for j in range(4):
            if i is not j:
                points[i].extend(get_points(xx[i], xx[j], n))
    points = [np.array(point) for point in points]
    return np.array(points)


def get_points(x1, x2, n):
    x2_mean = x2.mean()
    diff = np.linalg.norm(x1 - x2_mean, ord=2, axis=1)
    idxes = np.argsort(diff)
    return x1[idxes[:n:2]]


def plt_points(plt, x, s, alpha, c, fill=True):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    points_x = x[:, 0]
    points_y = x[:, 1]
    if fill:
        plt.scatter(points_x, points_y, s=s, alpha=alpha, c=colors[c])
    else:
        plt.scatter(points_x, points_y, s=s, alpha=alpha, edgecolors=colors[c], facecolors='none')
    return


def plt_lines(plt, x1, x2):
    for i in range(len(x1)):
        plt.plot([x1[i][0], x2[i][0]], [x1[i][1], x2[i][1]], linewidth=0.2, color='black')
    return


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def test(test_dataloader, model, adv='fgsm'):
    test_acc = 0
    train_n = 0
    for i, (x, y) in enumerate(test_dataloader):
        x = x.cuda()
        y = y.cuda()
        if adv == 'fgsm':
            x = fgsm(model, x, y)
        elif adv == 'pgd':
            x = pgd(model, x, y)
        out = model(x)

        test_acc += (out.max(1)[1] == y).sum().item()
        train_n += y.size(0)
    return test_acc / train_n
