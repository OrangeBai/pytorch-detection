import yaml
import torch
import torch.nn.functional as F
from config import *
import os


def evaluate_attack(model, attack_cls, test_loader):
    test_loss = 0
    test_acc = 0
    n = 0
    for i, (x, y) in enumerate(test_loader):
        x, y = x.cuda(), y.cuda()
        if attack_cls is not None:
            x = attack_cls.attack(x, y)
        with torch.no_grad():
            output = model(x)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n


def load_yml(file_path):
    with open(file_path) as file:
        config = yaml.full_load(file)
    return config


def get_nat_scheduler(args, yml_file):
    start_rate = yml_file['NAT'][args.NAT]['start_rate']
    end_epoch = yml_file['NAT'][args.NAT]['end_epoch']

    def nat_scheduler(epoch):
        if args.NAT == 'normal':
            return 1.01
        elif args.NAT == 'dynamic':
            return start_rate + (1 - start_rate) * epoch / args.epochs
        else:
            if epoch < end_epoch:
                return start_rate
            else:
                return 1.01

    return nat_scheduler


def init_delta(x, eps):
    delta = torch.zeros_like(x)
    for j in range(len(eps)):
        delta[:, j, :, :].uniform_(-eps[j][0][0].item(), eps[j][0][0].item())
    return delta


def clamp(x, lower_limit, upper_limit):
    return torch.max(torch.min(x, upper_limit), lower_limit)
