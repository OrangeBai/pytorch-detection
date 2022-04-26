import logging
import os
import datetime
import torchvision.models as models
import math
import torch
import yaml
from easydict import EasyDict
import shutil
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
from config import *


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(initial_lr, optimizer, epoch, n_repeats):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // int(math.ceil(30. / n_repeats))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fgsm(gradz, step_size):
    return step_size * torch.sign(gradz)


def accuracy(output, target, topk=(1,)):
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


def initiate_logger(output_path, evaluate):
    if not os.path.isdir(os.path.join('output', output_path)):
        os.makedirs(os.path.join('output', output_path))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join('output', output_path, 'eval.txt' if evaluate else 'log.txt'), 'w'))
    logger.info(pad_str(' LOGISTICS '))
    logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logger.info('Output Name: {}'.format(output_path))
    logger.info('User: {}'.format(os.getenv('USER')))
    return logger


def get_model_names():
    return sorted(name for name in models.__dict__
                  if name.islower() and not name.startswith("__")
                  and callable(models.__dict__[name]))


def pad_str(msg, total_len=70):
    rem_len = total_len - len(msg)
    return '*' * int(rem_len / 2) + msg + '*' * int(rem_len / 2)


def parse_config_file(args):
    with open(args.config) as f:
        config = EasyDict(yaml.load(f))

    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v

    # Add the output path
    config.output_name = '{:s}_step{:d}_eps{:d}_repeat{:d}'.format(args.output_prefix,
                                                                   int(config.ADV.fgsm_step), int(config.ADV.clip_eps),
                                                                   config.ADV.n_repeats)
    return config


def save_checkpoint(state, is_best, filepath, epoch):
    filename = os.path.join(filepath, f'checkpoint_epoch{epoch}.pth.tar')
    # Save model
    torch.save(state, filename)
    # Save best model
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))


class SkipErrorImageFolder(datasets.ImageFolder):
    __init__ = datasets.ImageFolder.__init__

    def __getitem__(self, index):
        try:
            return super(SkipErrorImageFolder, self).__getitem__(index)
        except Exception as e:
            print(e)

    def __len__(self):
        return len(self.samples) - 1


class SkipErrorDataloader(torch.utils.data.DataLoader):
    __init__ = torch.utils.data.DataLoader.__init__

    def __iter__(self):
        try:
            return super(SkipErrorDataloader, self).__iter__()
        except Exception as e:
            print(e)


def get_loader(configs, train_dir, val_dir):
    if configs.DATA.img_size > 0:
        transform = transforms.Compose([transforms.Resize(configs.DATA.img_size),
                                        transforms.RandomResizedCrop(configs.DATA.crop_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.RandomResizedCrop(configs.DATA.crop_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])

    train_dataset = SkipErrorImageFolder(train_dir, transform=transform)
    val_dataset = SkipErrorImageFolder(val_dir, transform=transform)

    train_loader = SkipErrorDataloader(
        train_dataset, batch_size=configs.DATA.batch_size, shuffle=True,
        num_workers=configs.DATA.workers, pin_memory=True, sampler=None)

    val_loader = SkipErrorDataloader(
        train_dataset, batch_size=configs.DATA.batch_size, shuffle=True,
        num_workers=configs.DATA.workers, pin_memory=True, sampler=None)
    return train_loader, val_loader
