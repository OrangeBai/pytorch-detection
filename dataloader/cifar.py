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


def get_loaders(args):
    mean, std = get_mean_std(args.dataset)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    num_workers = 2
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            DATA_PATH, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            DATA_PATH, train=False, transform=test_transform, download=True)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            DATA_PATH, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            DATA_PATH, train=False, transform=test_transform, download=True)
    else:
        raise NameError('No module called {0}'.format(args.dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader

