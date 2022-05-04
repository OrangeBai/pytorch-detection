import argparse
from config import *
import os
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', default='epoch', type=str, choices=['epoch', 'step'])
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_epoch', default=60, type=int)
    parser.add_argument('--total_step', default=60000, type=int)
    parser.add_argument('--lr_scheduler', default='milestones', choices=['static', 'milestones', 'exp', ',linear'])

    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--milestones', default=[0.5, 0.75])
    parser.add_argument('--num_workers', default=4)

    parser.add_argument('--net', default='VGG16', type=str)
    parser.add_argument('--print_every', default=100)

    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--mini', default=True, type=bool)

    parser.add_argument('--exp_id', default=0)

    return parser


def set_up_training(train=True):
    parser = get_args()
    args = parser.parse_args()

    parser.add_argument('--lr_step', default=lr_step(args), type=int, help='number of optimizer updates')
    parser.add_argument('--model_dir', default=name(args, train), type=str, help='model directory')
    parser.add_argument('--num_cls', default=num_cls(args), type=int, help='number of classes')
    parser.add_argument('--data_dir', default=data_dir(args), type=str, help='data dir')
    return parser.parse_args()


def lr_step(args):
    """
    :param args:
    :return: scheduler update steps
    """
    if args.train_type == 'epoch':
        return args.num_epoch
    else:
        return args.total_step


def name(args, train):
    exp_name = '_'.join([args.dataset, str(args.exp_id)])
    model_dir = os.path.join(MODEL_PATH, args.dataset, exp_name)
    if train:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)
    return model_dir


def num_cls(args):
    if args.dataset == 'cifar10':
        return 10
    else:
        return 100


def data_dir(args):
    if args.dataset == 'cifar10':
        return os.path.join(DATA_PATH, 'cifar10')
    else:
        return os.path.join(DATA_PATH, 'cifar100')
