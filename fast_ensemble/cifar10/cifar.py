import torch
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from config import *
import os


def get_mean_std(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    return mean, std


def get_limits(mean, std):
    mu = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()

    upper_limit = ((1 - mu) / std)
    lower_limit = ((0 - mu) / std)
    return upper_limit, lower_limit


def get_loaders(dir_, batch_size, dataset):
    mean, std = get_mean_std(dataset)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    num_workers = 2
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            dir_, train=False, transform=test_transform, download=True)
    else:
        raise NameError('No module called {0}'.format(dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


def set_up_exp(args):
    data_dir = os.path.join(DATA_PATH, args.dataset)
    if args.multiple_exp != 0:
        out_dir = os.path.join(MODEL_PATH, args.dataset, args.net + '_' + args.attack + '_' +
                               args.NAT + '_' + args.noise + '_' + str(args.epochs), str(args.exp_id))
    else:
        out_dir = os.path.join(MODEL_PATH, args.dataset, args.net + '_' + args.attack + '_' +
                               args.NAT + '_' + args.noise + '_' + str(args.epochs) + '_' + str(args.exp_id))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logfile = os.path.join(out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)
    model_path = os.path.join(out_dir, 'model.pth')
    return data_dir, out_dir, logfile, model_path
