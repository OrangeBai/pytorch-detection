import dataloader.cifar
import dataloader.MNIST
import dataloader.imagenet
from core.utils import *
import time
from copy import deepcopy


def set_loader(args):
    """
    Setting up data loader
    :param args:
    """
    if 'mnist' in args.dataset.lower():
        train_loader, test_loader = dataloader.MNIST.get_loaders(args)
    elif 'cifar' in args.dataset.lower():
        train_loader, test_loader = dataloader.cifar.get_loaders(args)
    elif 'imagenet' in args.dataset.lower():
        train_loader, test_loader = dataloader.imagenet.get_loaders(args)
    else:
        raise NameError()
    return train_loader, test_loader


def set_single_loaders(args, *labels):
    if 'mnist' in args.dataset.lower():
        return dataloader.MNIST.get_single_sets(args, *labels)
    elif 'cifar' in args.dataset:
        return dataloader.cifar.get_single_sets(args, *labels)


def set_mean_sed(args):
    if args.dataset == 'cifar10':
        mean, std = dataloader.cifar.CIAFR10_MEAN_STD
    elif args.dataset == 'cifar100':
        mean, std = dataloader.cifar.CIAFR100_MEAN_STD
    elif args.dataset == 'mnist':
        mean, std = dataloader.MNIST.MNIST_MEAN_STD
    else:
        raise NameError()
    return mean, std
