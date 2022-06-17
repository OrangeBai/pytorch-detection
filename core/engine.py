import logging

from torch.nn.functional import one_hot

from attack import *
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
    logging.info(warmup(model, InfiniteLoader(train_loader)))
    validate_model(model, -1, test_loader, True, alpha=1 / 255, eps=4 / 255, steps=7, restart=2)
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
        validate_model(model, -1, test_loader, True, alpha=1 / 255, eps=4 / 255, steps=7)

    model.save_model(args.model_dir)
    model.save_result(args.model_dir)
    return


def train_step(model, images, labels):
    images, labels = to_device(model.args.devices[0], images, labels)
    model.optimizer.zero_grad()
    outputs = model(images)
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
    model.optimizer.zero_grad()
    mean, std = set_mean_sed(model.args)

    noise_attack = Noise(model.model, model.args.devices[0], 4 / 255, mean=mean, std=std)
    lip = LipAttack(model.model, model.args.devices[0], eps=1 / 255, mean=mean, std=std)

    # float_hook = ModelHook(model, set_pattern_hook, Gamma=[0])
    # noised_sample = noise_attack.attack(images, 8, model.args.devices[0])
    # model.model(noised_sample)
    # float_neurons = float_hook.retrieve_res(retrieve_lb_ub, remove=True, sample_size=64)

    perturbation = lip.attack(images, labels)
    outputs = model(images)
    certified_res = model(images + perturbation) - outputs
    local_lip = (1 - one_hot(labels, num_classes=model.args.num_cls)).mul(certified_res).abs() * 10000 * 0.86
    if model.trained_ratio() < 0.3:
        rate = 1 / 4
    elif model.trained_ratio() < 0.6:
        rate = 2 / 4
    elif model.trained_ratio() < 0.8:
        rate = 3 / 4
    else:
        rate = 1
    loss = model.loss_function(outputs + rate * 100 * local_lip, labels)
    loss.backward()
    model.optimizer.step()
    model.lr_scheduler.step()

    top1, top5 = accuracy(outputs, labels)
    model.metrics.update(top1=(top1, len(images)), top5=(top5, len(images)), loss=(loss, len(images)),
                         lr=(model.optimizer.param_groups[0]['lr'], 1), aa=(local_lip.mean(), 1))
    model.metrics.synchronize_between_processes()
    return


def warmup(model, inf_loader):
    model.lr_scheduler = warmup_scheduler(model.args, model.optimizer)
    for cur_step in range(model.args.warmup_steps):
        images, labels = next(inf_loader)
        images, labels = to_device(model.args.devices[0], images, labels)
        cert_train_step(model, images, labels)
        if cur_step % model.args.print_every == 0:
            model.train_logging(cur_step, model.args.warmup_steps, -1, model.args.num_epoch, inf_loader.metric)

        if cur_step >= model.args.warmup_steps:
            break
    model.optimizer = init_optimizer(model.args, model.model)
    model.lr_scheduler = init_scheduler(model.args, model.optimizer)
    return


def validate_model(model, epoch, test_loader, robust=False, *args, **kwargs):
    start = time.time()
    model.eval()
    mean, std = set_mean_sed(model.args)
    if robust:
        fgsm = set_attack(model, 'FGSM', model.args.devices[0], mean=mean, std=std, *args, **kwargs)
        pgd = set_attack(model, 'PGD', model.args.devices[0], mean=mean, std=std, *args, **kwargs)
    for images, labels in test_loader:
        images, labels = to_device(model.args.devices[0], images, labels)
        pred = model.model(images)
        top1, top5 = accuracy(pred, labels)
        model.metrics.update(top1=(top1, len(images)), top5=(top5, len(images)))
        if robust:
            adv = fgsm.attack(images, labels)
            pred_adv = model.model(adv)
            fgsm_top1, fgsm_top5 = accuracy(pred_adv, labels)
            adv = pgd.attack(images, labels)
            pred_adv = model.model(adv)
            pgd_top1, pgd_top5 = accuracy(pred_adv, labels)
            model.metrics.update(fgsm_top1=(fgsm_top1, len(images)), fgsm_top5=(fgsm_top5, len(images)),
                                 pgd_top1=(pgd_top1, len(images)), pgd_top5=(pgd_top5, len(images)))

    model.train()
    msg = model.val_logging(epoch) + '\ttime:{0:.4f}'.format(time.time() - start)

    model.logger.info(msg)
    print(msg)
    return
