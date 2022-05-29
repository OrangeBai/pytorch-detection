import shutil
from config import *
import argparse
from dataloader.base import *
import sys


class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.unknown_args = []
        self.args = sys.argv[1:]
        self._init_parser()

    def _init_parser(self):
        # step-wise or epoch-wise
        self.parser.add_argument('--train_type', default='epoch', type=str, choices=['epoch', 'step'])
        self.parser.add_argument('--batch_size', default=64, type=int)
        self.parser.add_argument('--batch_norm', default=True, type=int)
        self.parser.add_argument('--reg', default=False, type=int)
        # scheduler and optimizer
        self.parser.add_argument('--lr_scheduler', default='linear', choices=['static', 'milestones', 'exp', 'linear'])
        self.parser.add_argument('--optimizer', default='Adam', choices=['SGD', 'Adam'])
        self.parser.add_argument('--lr', default=0.001, type=float)
        # attacks
        self.parser.add_argument('--attack', default='FGSM', type=str)
        # model type
        self.parser.add_argument('--model_type', default='mini', choices=['dnn', 'mini', 'nets'])
        self.parser.add_argument('--net', default='dnn', type=str)
        # training settings
        self.parser.add_argument('--num_workers', default=1)
        self.parser.add_argument('--print_every', default=50)
        # dataset and experiments
        self.parser.add_argument('--dataset', default='mnist', type=str)
        self.parser.add_argument('--exp_id', default=0, type=str)
        # gpu settings
        self.parser.add_argument('--cuda', default=[0], type=list)

        # for debugging
        self.parser.add_argument('--mode', default='client')
        self.parser.add_argument('--port', default=52162)

        self.train_type()
        self.lr_scheduler()
        self.optimizer()
        self.model_type()
        self.num_cls()
        return self.parser

    def get_args(self):
        return self.parser

    def train_type(self):
        """
        set up num_epoch, epoch_step and total step according to given train type and datasest
        !!!!! this must be called after dataset is assigned
        """
        args, _ = self.parser.parse_known_args(self.args)
        if args.train_type == 'epoch':
            train_loader, _ = set_loader(args)
            self.parser.add_argument('--num_epoch', default=60, type=int)
            self.parser.add_argument('--epoch_step', default=len(train_loader), type=int)
            args, _ = self.parser.parse_known_args(self.args)
            self.parser.add_argument('--total_step', default=args.num_epoch * args.epoch_step, type=int)
        else:
            self.parser.add_argument('--total_step', default=600, type=int)
            self.parser.add_argument('--epoch_step', default=100, type=int)
            args, _ = self.parser.parse_known_args(self.args)
            self.parser.add_argument('--num_epoch', default=args.total_step // args.epoch_step, type=int)
        return

    def lr_scheduler(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.lr_scheduler == 'milestones':
            self.parser.add_argument('--milestones', default=[0.5, 0.75])  # for milestone
        elif args.lr_scheduler == 'exp' or 'liner':
            self.parser.add_argument('--base_lr', default=0.001)  # for milestone
        elif args.lr_scheduler == 'cyclic':
            self.parser.add_argument('--base_lr', default=0.001)
            self.parser.add_argument('--up_ratio', default=1 / 3)
            self.parser.add_argument('--down_ratio', default=2 / 3)
        else:
            raise NameError('Scheduler {} not found'.format(args.lr_scheduler))
        return

    def optimizer(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.optimizer == 'SGD':
            # SGD parameters
            self.parser.add_argument('--weight_decay', default=5e-4, type=float)
            self.parser.add_argument('--momentum', default=0.9, type=float)
        elif args.optimizer == 'Adam':
            self.parser.add_argument('--beta_1', default=0.9, type=float)
            self.parser.add_argument('--beta_2', default=0.99, type=float)
            self.parser.add_argument('--eps', default=1e-8, type=float)
            self.parser.add_argument('--weight_decay', default=5e-4, type=float)
        else:
            pass
        return

    def model_type(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.net == 'dnn':
            self.parser.set_defaults(model_type='dnn')
            self.parser.add_argument('--input_size', default=784, type=int)
            self.parser.add_argument('--width', default=1000, type=int)
            self.parser.add_argument('--depth', default=9, type=int)
        else:
            pass
        return

    def num_cls(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.dataset.lower() == 'cifar10' or 'mnist':
            self.parser.add_argument('--num_cls', default=10)
        else:
            self.parser.add_argument('--num_cls', default=100)
        return


def devices(parser):
    """
        Check devices, notice that this function should be called separately for train and test
        to avoid conflicts caused by different device difference
    @param parser:
    @return:
    """
    args, _ = parser.parse_known_args()
    device_num = torch.cuda.device_count()
    if device_num == 0:
        parser.add_argument('--devices', default=[None])
    else:
        parser.add_argument('--devices', default=[d for d in args.cuda if d < device_num])
    return parser


def model_dir(parser, train):
    """
    set up the name of experiment: dataset_net_exp_id
    @param parser: current argument
    @param train:
    @return:
    """
    args, _ = parser.parse_known_args()
    exp_name = '_'.join([args.dataset, str(args.net), str(args.exp_id)])
    path = os.path.join(MODEL_PATH, args.dataset, exp_name)
    if train:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    parser.add_argument('--model_dir', default=path, type=str, help='model directory')
    return parser
