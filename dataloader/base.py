import dataloader.cifar
from core.utils import *


def set_loader(args):
    """
    Setting up data loader
    :param args:
    """
    if 'cifar' in args.dataset:
        train_loader, test_loader = dataloader.cifar.get_loaders(args)
    elif 'imagenet' in args.dataset:
        train_loader, test_loader = None, None
    else:
        raise NameError()
    return train_loader, test_loader


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
        self.metric.update(iter_time=(time.time() - self.last_time,  1))
        self.last_time = time.time()

        while True:
            try:
                obj = next(self.data_loader)
                self.metric.update(data_time=(time.time() - self.last_time,  1))

                self.metric.synchronize_between_processes()
                return obj
            except StopIteration:
                self.data_loader = iter(self.iterable)

    def pack_metric(self, reset=False):
        """
        return a metric object.
        @param reset: Reset the time loader
        @return:
        """
        if reset:
            self.metric.reset()
        return self.metric
