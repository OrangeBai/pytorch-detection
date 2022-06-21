import shutil
from config import *
import argparse
from dataloader.base import *
import sys
import os
import yaml
import torch


class ArgParser:
    def __init__(self, train=False, argv=None):
        self.parser = argparse.ArgumentParser()
        self.unknown_args = []
        if argv is None:
            self.args = sys.argv[1:]
        else:
            self.args = sys.argv[1:] + argv
        self._init_parser()
        self.model_dir(train)
        self.devices()
        if train:
            self.save()
        else:
            path = os.path.join(self.get_args().model_dir, 'args.yaml')
            self.modify_parser(path)

        self.files = self.set_files()

    def _init_parser(self):
        self.parser.add_argument('--resume', default=False, action='store_true')
        # step-wise or epoch-wise
        self.parser.add_argument('--epoch_type', default='epoch', type=str, choices=['epoch', 'step'])
        self.parser.add_argument('--batch_size', default=128, type=int)

        # model settings
        self.parser.add_argument('--batch_norm', default=True, type=int)
        self.parser.add_argument('--activation', default='PReLU', type=str)
        # trainer settings
        self.parser.add_argument('--train_mode', default='normal', type=str)
        self.parser.add_argument('--val_mode', default='normal', type=str)
        # scheduler and optimizer
        self.parser.add_argument('--lr_scheduler', default='milestones',
                                 choices=['static', 'milestones', 'exp', 'linear', 'cyclic'])
        self.parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'])
        self.parser.add_argument('--lr', default=0.1, type=float)
        self.parser.add_argument('--warmup', default=2, type=float)
        # attacks
        self.parser.add_argument('--attack', default='vanila', type=str, choices=['vanila', 'fgsm', 'pgd', 'ffgsm'])
        # model type
        self.parser.add_argument('--model_type', default='net', choices=['dnn', 'mini', 'net'])
        self.parser.add_argument('--net', default='vgg16', type=str)
        # training settings
        self.parser.add_argument('--num_workers', default=1, type=int)
        self.parser.add_argument('--print_every', default=50, type=int)

        #
        self.parser.add_argument('--config', default=None)
        self.parser.add_argument('--prune_ratio', default=0.95)

        # dataset and experiments
        self.parser.add_argument('--dataset', default='cifar100', type=str)
        self.parser.add_argument('--exp_id', default=0, type=str)
        # gpu settings
        self.parser.add_argument('--cuda', default=[0], type=list)
        # for debugging
        self.parser.add_argument('--mode', default='client')
        self.parser.add_argument('--port', default=52162)

        self.epoch_type()
        self.lr_scheduler()
        self.optimizer()
        self.model_type()
        self.dataset()
        self.train_mode()
        return self.parser

    def get_args(self):
        return self.parser.parse_args(self.args)

    def resume(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.resume:
            self.parser.add_argument('--resume_name', default=None)
        return

    def epoch_type(self):
        """
        set up num_epoch, epoch_step and total step according to given train type and datasest
        !!!!! this must be called after dataset is assigned
        """
        args, _ = self.parser.parse_known_args(self.args)
        if args.epoch_type == 'epoch':
            train_loader, _ = set_loader(args)
            self.parser.add_argument('--num_epoch', default=200, type=int)
            self.parser.add_argument('--epoch_step', default=len(train_loader), type=int)
            self.parser.add_argument('--warmup_steps', default=int(len(train_loader) * args.warmup), type=int)
            args, _ = self.parser.parse_known_args(self.args)
            self.parser.add_argument('--total_step', default=args.num_epoch * args.epoch_step, type=int)
        else:
            self.parser.add_argument('--total_step', default=600, type=int)
            self.parser.add_argument('--epoch_step', default=100, type=int)
            args, _ = self.parser.parse_known_args(self.args)
            self.parser.add_argument('--num_epoch', default=args.total_step // args.epoch_step, type=int)
            self.parser.add_argument('--warmup_steps', default=int(args.epoch_step * args.warmup), type=int)
        return

    def lr_scheduler(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.lr_scheduler == 'milestones':
            self.parser.add_argument('--gamma', default=0.2, type=float)
            self.parser.add_argument('--milestones', default=[0.3, 0.6, 0.8], nargs='+', type=float)  # for milestone
        elif args.lr_scheduler in ['exp', 'linear']:
            self.parser.add_argument('--base_lr', default=0.001 * args.lr)  # for linear
        elif args.lr_scheduler == 'cyclic':
            self.parser.add_argument('--base_lr', default=0.001 * args.lr)
            self.parser.add_argument('--up_ratio', default=1 / 40)
            self.parser.add_argument('--down_ratio', default=2 / 40)
        else:
            raise NameError('Scheduler {} not found'.format(args.lr_scheduler))
        return

    def optimizer(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.optimizer == 'SGD':
            # SGD parameters
            self.parser.set_defaults(lr=0.1)
            self.parser.add_argument('--weight_decay', default=5e-4, type=float)
            self.parser.add_argument('--momentum', default=0.9, type=float)
        elif args.optimizer == 'Adam':
            self.parser.set_defaults(lr=0.01)
            self.parser.add_argument('--beta_1', default=0.9, type=float)
            self.parser.add_argument('--beta_2', default=0.99, type=float)
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
        elif 'cxfy' in args.net.lower():
            self.parser.set_defaults(model_type='net')
            self.parser.add_argument('--shape', default='large', type=str)
        return

    def dataset(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.dataset.lower() == 'mnist':
            self.parser.add_argument('--num_cls', default=10)
        elif args.dataset.lower() == 'cifar10':
            self.parser.add_argument('--num_cls', default=10)
            self.parser.set_defaults(model_type='mini')
        elif args.dataset.lower() == 'cifar100':
            self.parser.add_argument('--num_cls', default=100)
            self.parser.set_defaults(model_type='mini')
        elif args.dataset.lower() == 'imagenet':
            self.parser.add_argument('--num_cls', default=1000)
            self.parser.set_defaults(model_type='net')
        return

    def devices(self):
        """
            Check devices, notice that this function should be called separately for train and test
            to avoid conflicts caused by different device difference
        @return:
        """
        args, _ = self.parser.parse_known_args(self.args)
        device_num = torch.cuda.device_count()
        if device_num == 0:
            self.parser.add_argument('--devices', default=[None])
        else:
            self.parser.add_argument('--devices', default=[d for d in args.cuda if d < device_num])
        return self.parser

    def model_dir(self, train):
        """
        set up the name of experiment: dataset_net_exp_id
        @param train:
        @return:
        """
        args, _ = self.parser.parse_known_args(self.args)
        exp_name = '_'.join([args.dataset, str(args.net), str(args.exp_id)])
        path = os.path.join(MODEL_PATH, args.dataset, exp_name)
        if train:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        self.parser.add_argument('--model_dir', default=path, type=str, help='model directory')
        return self.parser

    def modify_parser(self, file_path):
        cur_args, _ = self.parser.parse_known_args(self.args)
        # Load configuration from yaml file
        with open(file_path, 'r') as file:
            args_dict = yaml.load(file, Loader=yaml.FullLoader)

        for key, val in args_dict.items():
            if key not in vars(cur_args).keys():
                self.parser.add_argument('--' + key, default=val, type=type(val))
        return

    def save(self):
        args = self.parser.parse_args(self.args)
        json_file = os.path.join(args.model_dir, 'args.yaml')
        args_dict = vars(args)
        with open(json_file, 'w') as f:
            yaml.dump(args_dict, f)
        return

    def set_files(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.dataset.lower() == 'imagenet':
            self.parser.add_argument('--yaml_files', default='default', type=str)

            return os.listdir(os.path.join(os.getcwd(), self.get_args().yaml_files))
        else:
            args = self.parser.parse_args()
            return [os.path.join(args.model_dir, 'args.yaml')]

    def train_mode(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.train_mode == 'normal':
            pass
        elif args.train_mode == 'cert':
            self.parser.add_argument('--fre_lip_est', default=50, type=int)
            self.parser.add_argument('--num_flt_est', default=64, type=int)
            self.parser.add_argument('--noise_eps', default=2 / 255, type=float)
            self.parser.add_argument('--alpha', default=16 / 255, type=float)
            self.parser.add_argument('--eps', default=32 / 255, type=float)
            self.parser.add_argument('--ord', default='l2', type=str)
