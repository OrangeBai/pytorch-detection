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
        return mnist.get_single_sets(args, *labels)
    elif 'cifar' in args.dataset:
        return cifar.get_single_sets(args, *labels)

