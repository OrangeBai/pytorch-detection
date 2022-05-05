import argparse
import torch
from config import *
import os
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    # step-wise or epoch-wise
    parser.add_argument('--train_type', default='epoch', type=str, choices=['epoch', 'step'])
    parser.add_argument('--num_epoch', default=60, type=int)
    parser.add_argument('--total_step', default=600, type=int)
    parser.add_argument('--epoch_step', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    # scheduler settings
    parser.add_argument('--lr_scheduler', default='linear', choices=['static', 'milestones', 'exp', 'linear'])
    parser.add_argument('--milestones', default=[0.5, 0.75])  # for milestone

    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'])
    # SGD parameters
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)

    # attacks
    parser.add_argument('--attack', default='FGSM', type=str)

    parser.add_argument('--num_workers', default=1)
    parser.add_argument('--model_type', default='mini', choices=['dnn', 'mini', 'nets'])
    parser.add_argument('--net', default='vgg16', type=str)
    parser.add_argument('--print_every', default=50)

    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--exp_id', default=0)

    parser.add_argument('--cuda', default=[0], type=list)

    # for debugging
    parser.add_argument('--mode', default='client')
    parser.add_argument('--port', default=52162)

    return parser


def set_up_training(train=True):
    parser = get_args()
    args = parser.parse_args()

    parser.add_argument('--model_dir', default=name(args, train), type=str, help='model directory')
    parser.add_argument('--num_cls', default=num_cls(args), type=int, help='number of classes')
    parser.add_argument('--data_dir', default=data_dir(args), type=str, help='data dir')
    parser.add_argument('--devices', default=check_cuda(args), type=list, help='check gpu devices')
    return parser.parse_args()


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


def check_cuda(args):
    device_num = torch.cuda.device_count()
    if device_num == 0:
        return [None]
    else:
        return [d for d in args.cuda if d < device_num]


def data_dir(args):
    if args.dataset == 'cifar10':
        return os.path.join(DATA_PATH, 'cifar10')
    else:
        return os.path.join(DATA_PATH, 'cifar100')
