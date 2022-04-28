import dataloader.cifar
from core.utils import *


def set_loader(args):
    if 'cifar' in args.dataset:
        train_loader, test_loader = dataloader.cifar.get_loaders(args)
    elif 'imagenet' in args.dataset:
        train_loader, test_loader = None, None
    else:
        raise NameError()
    return train_loader, test_loader


class InfiniteLoader:
    def __init__(self, iterable):
        self.iterable = iterable
        self.data_loader = iter(self.iterable)
        self.counter = 0

        self.last_time = time.time()
        self.data_time = SmoothedValue()
        self.iter_time = SmoothedValue()

    def __iter__(self):
        return self

    def __next__(self):
        self.iter_time.update(time.time() - self.last_time)
        self.last_time = time.time()

        while True:
            try:
                obj = next(self.data_loader)
                self.data_time.update(time.time() - self.last_time)
                self.data_time.synchronize_between_processes()
                self.iter_time.synchronize_between_processes()
                return obj
            except StopIteration:
                self.data_loader = iter(self.iterable)

    def pack_metric(self, reset=False):
        packed = {'iter_time': self.iter_time, 'data_time': self.data_time}
        if reset:
            self.iter_time = SmoothedValue()
            self.data_time = SmoothedValue()
        return packed
