import torch
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from config import *
import numpy as np


def get_mean_std(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    return mean, std


def get_loaders(dir_, batch_size, dataset):
    mean, std = get_mean_std(dataset)
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
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def get_test_set(*args):
    """
    Load test dataset according to labels:
    'all': all data
    1 : data with label 1
    2 : data with label 2 ......
    :param args: labels:
                    'all': all data
                    1 : data with label 1
                    2 : data with label 2 ......
    :return: a collection of data loaders
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_loaders = []
    test_sets = []
    for arg in args:
        if arg == 'all':
            test_dataset = datasets.MNIST(DATA_PATH, train=False, transform=test_transform, download=True)
        elif arg in range(9):
            test_dataset = datasets.MNIST(DATA_PATH, train=False, transform=test_transform, download=True)
            test_dataset.data = test_dataset.data[np.where(test_dataset.targets.numpy() == arg)][:1000]
            test_dataset.targets = test_dataset.targets[np.where(test_dataset.targets.numpy() == arg)][:1000]
        else:
            raise ValueError("Cannot find dataset %s" % str(args))

        test_sets += [test_dataset]
        test_loaders += [torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=128,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )]
    return test_loaders, test_sets
