import dataloader.cifar
import dataloader.MNIST
import dataloader.imagenet
import time
from core.utils import MetricLogger


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
    if args.dataset.lower() == 'cifar10':
        mean, std = dataloader.cifar.CIAFR10_MEAN_STD
    elif args.dataset.lower() == 'cifar100':
        mean, std = dataloader.cifar.CIAFR100_MEAN_STD
    elif args.dataset.lower() == 'mnist':
        mean, std = dataloader.MNIST.MNIST_MEAN_STD
    else:
        raise NameError()
    return mean, std


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
