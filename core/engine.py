import logging

from attack import *
from core.pattern import *
from dataloader.base import *
from models import *


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

            # model.train_step(images, labels)
            cert_train_step(model, images, labels)
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


def train_step(model, images, labels):
    images, labels = to_device(model.args.devices[0], images, labels)
    model.optimizer.zero_grad()
    outputs = model.model(images)
    loss = model.loss_function(outputs, labels)
    loss.backward()
    model.optimizer.step()
    model.lr_scheduler.step()
    top1, top5 = accuracy(outputs, labels)
    model.metrics.update(top1=(top1, len(images)), top5=(top5, len(images)), loss=(loss, len(images)),
                         lr=(model.optimizer.param_groups[0]['lr'], 1))
    model.metrics.synchronize_between_processes()
    return


def cert_train_step(model, images, labels):
    images, labels = to_device(model.args.devices[0], images, labels)
    mean, std = set_mean_sed(model.args)

    noise_attack = Noise(model.model, model.args.devices[0], 4 / 255, mean=mean, std=std)
    lip = LipAttack(model.model, model.args.devices[0], eps=1 / 255, mean=mean, std=std)

    # float_hook = ModelHook(model, set_pattern_hook, Gamma=[0])
    # noised_sample = noise_attack.attack(images, 8, model.args.devices[0])
    # model.model(noised_sample)
    # float_hook.retrieve_res(retrieve_float_neurons, remove=True, sample_size=64)

    perturbation = lip.attack(images, labels)
    certified_res = (model.model(images + perturbation) - model.model(images)) * 10
    aa = (1 - torch.nn.functional.one_hot(labels)).mul(certified_res).abs()

    outputs = model.model(images)
    loss = model.loss_function(outputs + aa.abs(), labels)
    loss.backward()
    model.optimizer.step()
    model.lr_scheduler.step()

    top1, top5 = accuracy(outputs, labels)
    model.metrics.update(top1=(top1, len(images)), top5=(top5, len(images)), loss=(loss, len(images)),
                         lr=(model.optimizer.param_groups[0]['lr'], 1))
    model.metrics.synchronize_between_processes()
    return