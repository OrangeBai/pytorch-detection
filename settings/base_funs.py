import os
import shutil
import torch
from config import *
import argparse
from dataloader.base import *


def get_args():
    parser = argparse.ArgumentParser()
    # step-wise or epoch-wise
    parser.add_argument('--train_type', default='epoch', type=str, choices=['epoch', 'step'])
    parser.add_argument('--batch_size', default=128, type=int)
    # scheduler and optimizer
    parser.add_argument('--lr_scheduler', default='linear', choices=['static', 'milestones', 'exp', 'linear'])
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--lr', default=0.05, type=float)
    # attacks
    parser.add_argument('--attack', default='FGSM', type=str)
    # model type
    parser.add_argument('--model_type', default='dnn', choices=['dnn', 'mini', 'nets'])
    parser.add_argument('--net', default='dnn', type=str)
    # training settings
    parser.add_argument('--num_workers', default=1)
    parser.add_argument('--print_every', default=50)
    # dataset and experiments
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--exp_id', default=1)
    # gpu settings
    parser.add_argument('--cuda', default=[0], type=list)

    # To make the script tidy and aligned, we input the parser to each functions
    parser = train_type(parser)  # set up num_epoch, total_step and epoch_step
    parser = lr_scheduler(parser)
    parser = optimizer(parser)
    parser = model_type(parser)
    parser = num_cls(parser)
    # for debugging
    parser.add_argument('--mode', default='client')
    parser.add_argument('--port', default=52162)

    return parser


def model_dir(parser, train):
    """
    set up the name of experiment: dataset_net_exp_id
    @param parser: current argument
    @param train:
    @return:
    """
    args = parser.parse_args()
    exp_name = '_'.join([args.dataset, str(args.net), str(args.exp_id)])
    path = os.path.join(MODEL_PATH, args.dataset, exp_name)
    if train:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    parser.add_argument('--model_dir', default=path, type=str, help='model directory')
    return parser


def train_type(parser):
    """
    set up num_epoch, epoch_step and total step according to given train type and datasest
    !!!!! this must be called after dataset is assigned
    @param parser:
    @return:
    """
    args = parser.parse_args()
    if args.train_type == 'epoch':
        train_loader = set_loader(args)
        parser.add_argument('--num_epoch', default=60, type=int)
        parser.add_argument('--epoch_step', default=len(train_loader), type=int)
        args = parser.parse_args()
        parser.add_argument('--total_step', default=args.num_epoch * args.epoch_step, type=int)
    else:
        parser.add_argument('--total_step', default=600, type=int)
        parser.add_argument('--epoch_step', default=100, type=int)
        args = parser.parse_args()
        parser.add_argument('--num_epoch', default=args.total_step // args.epoch_step, type=int)
    return parser


def lr_scheduler(parser):
    args = parser.parse_args()
    if args.lr_scheduler == 'milestones':
        parser.add_argument('--milestones', default=[0.5, 0.75])  # for milestone
    elif args.lr_scheduler == 'exp' or 'liner' or 'cyclic':
        parser.add_argument('--base_lr', default=0.01, )  # for milestone
    return parser


def optimizer(parser):
    args = parser.parse_args()
    if args.optimizer == 'SGD':
        # SGD parameters
        parser.add_argument('--weight_decay', default=5e-4, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
    else:
        pass
    return parser


def model_type(parser):
    args = parser.parse_args()
    if args.model_type == 'dnn':
        parser.add_argument('--input_size', default=784, type=int)
        parser.add_argument('--width', default=100, type=int)
        parser.add_argument('--depth', default=9, type=int)
    else:
        pass
    return parser


def num_cls(parser):
    args = parser.parse_args()
    if args.dataset.lower() == 'cifar10' or 'mnist':
        parser.add_argument('--num_cls', default=10)
    else:
        parser.add_argument('--num_cls', default=100)
    return parser


def devices(parser):
    """
        Check devices, notice that this function should be called separately for train and test
        to avoid conflicts caused by different device difference
    @param parser:
    @return:
    """
    args = parser.parse_args()
    device_num = torch.cuda.device_count()
    if device_num == 0:
        parser.add_argument('--device_num', default=[None])
    else:
        parser.add_argument('--device', default=[d for d in args.cuda if d < device_num])
    return parser
