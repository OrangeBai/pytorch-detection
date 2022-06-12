import time
from core.utils import SmoothedValue
import logging
from models import *
from settings.train_settings import *


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


def train_model(args):
    train_loader, test_loader = set_loader(args)

    model = BaseModel(args)
    logging.info(model.warmup(InfiniteLoader(train_loader)))
    inf_loader = InfiniteLoader(train_loader)

    for cur_epoch in range(args.num_epoch):
        for cur_step in range(args.epoch_step):
            images, labels = next(inf_loader)
            model.train_step(images, labels)

            if cur_step % args.print_every == 0:
                model.train_logging(cur_step, args.epoch_step, cur_epoch, args.num_epoch, inf_loader.metric)

        model.epoch_logging(cur_epoch, args.num_epoch, time_metrics=inf_loader.metric)
        inf_loader.reset()
        # if cur_epoch % 10 == 0 and cur_epoch != 0:
        #     model.pruning_val(cur_epoch, test_loader)
        # else:
        model.validate_model(cur_epoch, test_loader)

    model.save_model(args.model_dir)
    model.save_result(args.model_dir)
    return


