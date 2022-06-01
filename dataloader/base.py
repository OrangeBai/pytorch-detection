import dataloader.cifar as cifar
import dataloader.MNIST as mnist
from core.utils import *
import time
from copy import deepcopy


def set_loader(args):
    """
    Setting up data loader
    :param args:
    """
    if 'mnist' in args.dataset.lower():
        train_loader, test_loader = mnist.get_loaders(args)
    elif 'cifar' in args.dataset:
        train_loader, test_loader = cifar.get_loaders(args)
    elif 'imagenet' in args.dataset:
        train_loader, test_loader = None, None
    else:
        raise NameError()
    return train_loader, test_loader


def set_single_loaders(args, *labels):
    if 'mnist' in args.dataset.lower():
        return mnist.get_single_sets(args, *labels)
    elif 'cifar' in args.dataset:
        return cifar.get_single_sets(args, *labels)


class InfiniteLoader:
    def __init__(self, iterable):
        """
        Initializer
        @param iterable: An Dataset object
        """
        self.iterable = iterable
        self.data_loader = iter(self.iterable)
        self.counter = 0

        self.last_time = time.time()
        self.metric = MetricLogger()

    def __iter__(self):
        return self

    def __next__(self):
        self.metric.update(iter_time=(time.time() - self.last_time, 1))
        self.last_time = time.time()

        while True:
            try:
                obj = next(self.data_loader)
                self.metric.update(data_time=(time.time() - self.last_time, 1))

                self.metric.synchronize_between_processes()
                return obj
            except StopIteration:
                self.data_loader = iter(self.iterable)

    def reset(self):
        self.metric = MetricLogger()
