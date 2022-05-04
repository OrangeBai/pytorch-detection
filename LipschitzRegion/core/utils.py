import torch
from collections import defaultdict, deque
import torch.distributed as dist
import time
import datetime as datetime
import math
import numpy as np
import os


def init_scheduler(args, optimizer):
    if args.lr_scheduler == 'milestones':
        milestones = [milestone * args.lr_step for milestone in args.milestones]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.lr_scheduler == 'static':
        def lambda_rule(t):
            return 1.0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_scheduler == 'exp':
        gamma = math.pow(1 / 100, 1 / args.num_epoch)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    elif args.lr_scheduler == 'cycle':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,
                                                         step_size_up=args.lr_step / 3,
                                                         step_size_down=2 * args.lr_step / 3)
    elif args.lr_scheduler == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1,
                                                         step_size_up=1,
                                                         step_size_down=args.lr_step)
    else:
        raise NameError('Scheduler {0} not found'.format(args.lr_scheduler))
    return lr_scheduler


def init_optimizer(args, model):
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise NameError('Optimizer {0} not found'.format(args.lr_scheduler))
    return optimizer


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{avg:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / (self.count + 1e-2)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.iter_time = SmoothedValue(fmt='{global_avg:.4f}')

    def reset(self):
        for name in self.meters.keys():
            self.meters[name].reset()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def update(self, **kwargs):
        for k, (v, n) in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def epoch_logger(self, iterable):
        end = time.time()
        for idx, obj in enumerate(iterable):
            yield idx, obj
            self.iter_time.update(time.time() - end)
            end = time.time()

    def infinite_logger(self, iterable):
        end = time.time()
        counter = 0
        iter_loader = iter(iterable)
        while True:
            try:
                obj = next(iter_loader)
                yield counter, obj
                counter += 1
                self.iter_time.update(time.time() - end)
                end = time.time()
            except StopIteration:
                iter_loader = iter(iterable)

    def logging(self, step, batch_num, epoch=None, epoch_num=None):
        space_fmt = ':' + str(len(str(batch_num))) + 'd'

        log_msg = '  '.join(['[{0' + space_fmt + '}/{1}]',
                             'eta: {eta}',
                             '{meters}',
                             'time: {time:.4f}',
                             'max mem: {memory:.2f}'
                             ])
        eta_seconds = self.iter_time.global_avg * (batch_num - step)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        msg = log_msg.format(
            step, batch_num, eta=eta_string,
            meters=str(self),
            time=self.iter_time.global_avg,
            # data=str(self.data_time),
            memory=torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        )

        if epoch_num is not None and epoch is not None:
            msg = 'Epoch: [{epoch}/{epoch_num}]'.format(epoch=epoch, epoch_num=epoch_num) + msg

        return msg


def save_model(model, path, name=None):
    if not name:
        model_path = os.path.join(path, 'weights.pth')
    else:
        model_path = os.path.join(path, 'weights_{}.pth'.format(name))
    torch.save(model.state_dict(), model_path)
    return


def load_model(model, path, name=None):
    if not name:
        model_path = os.path.join(path, 'weights.pth')
    else:
        model_path = os.path.join(path, 'weights_{}.pth'.format(name))
    model.load_state_dict(torch.load(model_path), strict=False)
    return


def save_result(res, path, name=None):
    if not name:
        res_path = os.path.join(path, 'result')
    else:
        res_path = os.path.join(path, 'result_{}'.format(name))
    np.save(res_path, res)


def load_result(path, name=None):
    if not name:
        res_path = os.path.join(path, 'result')
    else:
        res_path = os.path.join(path, 'result_{}'.format(name))
    return np.load(res_path)

