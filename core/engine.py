import time
from core.utils import SmoothedValue


class InfiniteLoader:
    def __init__(self, iterable):
        self.iterable = iterable
        self.data_loader = iter(self.iterable)
        self.counter = 0
        self.iter_time = SmoothedValue()

    def __iter__(self):
        return self

    def __next__(self):
        end = time.time()

        while True:
            try:
                obj = next(self.data_loader)
                self.iter_time.update(time.time() - end)
                return obj
            except StopIteration:
                self.data_loader = iter(self.iterable)

    def retrieve_time(self):
        return str(self.iter_time)


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5]
    b = InfiniteLoader(a)
    for i in b:
        print(i)
        if i > 100:
            break
