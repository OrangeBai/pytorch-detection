import torch.nn as nn
from core.utils import *
import importlib
from core.pattern import *
import os
import time
import datetime
import logging
from attack import *
from collections import OrderedDict
from core.prune import *

class BaseModel(nn.Module):
    # TODO Record epoch info
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.model = build_model(args)
        self.optimizer = init_optimizer(args, self.model)
        self.lr_scheduler = init_scheduler(args, self.optimizer)
        self.loss_function = self.set_loss()

        self.result = {'train': dict(), 'test': dict()}
        self.metrics = MetricLogger()
        # self.attack = set_attack(self.model, args.attack, )

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=os.path.join(args.model_dir, 'logger'))
        self.logger.info(args)

        if args.resume:
            self.load_model(args.model_dir, args.resume_name)

    @staticmethod
    def set_loss():
        # TODO add more losses
        return nn.CrossEntropyLoss()

    def save_model(self, path, name=None):
        if not name:
            model_path = os.path.join(path, 'weights.pth')
        else:
            model_path = os.path.join(path, 'weights_{}.pth'.format(name))
        torch.save(self.model.state_dict(), model_path)
        return

    def load_model(self, path, name=None):
        if not name:
            model_path = os.path.join(path, 'weights.pth')
        else:
            model_path = os.path.join(path, 'weights_{}.pth'.format(name))
        self.load_weights(torch.load(model_path))

        print('Loading model from {}'.format(model_path))
        return

    def load_weights(self, state_dict):
        new_dict = OrderedDict()
        for (k1, v1), (k2, v2) in zip(self.model.state_dict().items(), state_dict.items()):
            if v1.shape == v2.shape:
                new_dict[k1] = v2
            else:
                raise KeyError
        self.model.load_state_dict(new_dict)

    def save_result(self, path, name=None):
        if not name:
            res_path = os.path.join(path, 'result')
        else:
            res_path = os.path.join(path, 'result_{}'.format(name))
        np.save(res_path, self.result)

    def train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()


        top1, top5 = accuracy(outputs, labels)
        self.metrics.update(top1=(top1, len(images)), top5=(top5, len(images)), loss=(loss, len(images)),
                            lr=(self.optimizer.param_groups[0]['lr'], 1))
        self.metrics.synchronize_between_processes()
        return

    def record_result(self, epoch, mode='train'):

        epoch_result = {}
        for k, v in self.metrics.meters.items():
            epoch_result[k] = v.to_dict()
        self.result[mode][epoch] = epoch_result
        self.metrics.reset()
        return

    def validate_model(self, epoch, test_loader):
        start = time.time()
        self.model.eval()
        for images, labels in test_loader:
            images, labels = to_device(self.args.devices[0], images, labels)
            pred = self.model(images)
            top1, top5 = accuracy(pred, labels)
            self.metrics.update(top1=(top1, len(images)), top5=(top5, len(images)))

        self.model.train()
        msg = self.val_logging(epoch) + '\ttime:{0:.4f}'.format(time.time() - start)
        self.logger.info(msg)
        print(msg)
        return

    def pruning_val(self, epoch, test_loader):
        start = time.time()
        self.model.eval()
        ap_hook = ModelHook(self.model, set_input_hook)
        metric = MetricLogger()
        storage = []
        net_same_all = []
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = to_device(self.args.devices[0], images, labels)
            pre_ori = self.model(images)
            top1, top5 = accuracy(pre_ori, labels)
            unpacked = ap_hook.retrieve_res(unpack)
            metric.update(top1_avd=(top1, self.args.batch_size), top5_adv=(top5, self.args.batch_size))
            if idx == 0:
                net_same_all = find_dead_neuron(unpacked, [0])
            else:
                net_same_all = compute_mean(net_same_all, find_dead_neuron(unpacked, [0]), idx)

        net_same_all.insert(0, [])
        self.model = self.model.cpu()
        block_counter = 1
        new_model = []
        new_size = []
        for n, m in self.model.named_modules():
            if type(m) in [LinearBlock, ConvBlock, BasicBlock, BottleNeck]:
                cur_block_ps = net_same_all[block_counter]
                pre_block_ps = net_same_all[block_counter - 1]
                m, shape = prune_block(m, cur_block_ps, pre_block_ps, 0.98)
                new_model += [m]
                new_size += [shape]
                block_counter += 1
            if type(m) == nn.MaxPool2d:
                new_size += ['M']
        wt = self.model.state_dict()
        self.args.config = new_size
        self.model = build_model(self.args)
        self.load_weights(wt)

        self.model.train()
        self.model = self.model.cuda()
        self.optimizer.param_groups[0]['params'].clear()
        self.optimizer.param_groups[0]['params'].append(self.model.parameters())
        msg = self.val_logging(epoch) + '\ttime:{0:.4f}'.format(time.time() - start)
        self.logger.info(msg)

        print(msg)
        print('Triming to {0}'.format(new_size))
        self.logger.info('Triming to {0}'.format(new_size))
        return

    def warmup(self, inf_loader):
        self.lr_scheduler = warmup_scheduler(self.args, self.optimizer)
        for cur_step in range(self.args.warmup_steps):
            images, labels = next(inf_loader)
            images, labels = to_device(self.args.devices[0], images, labels)
            self.train_step(images, labels)
            if cur_step % self.args.print_every == 0:
                self.train_logging(cur_step, self.args.warmup_steps, -1, self.args.num_epoch, inf_loader.metric)

            if cur_step >= self.args.warmup_steps:
                break
        self.optimizer = init_optimizer(self.args, self.model)
        self.lr_scheduler = init_scheduler(self.args, self.optimizer)
        return

    def train_logging(self, step, batch_num, epoch, epoch_num, time_metrics=None):
        # TODO maybe a refactor???
        space_fmt = ':' + str(len(str(batch_num))) + 'd'

        log_msg = '  '.join(['Epoch: [{epoch}/{epoch_num}]',
                             '[{step' + space_fmt + '}/{batch_num}]',
                             '{time_str}',
                             '{meters}',
                             '{memory}'
                             ])

        if time_metrics is not None:
            eta_seconds = time_metrics.meters['iter_time'].global_avg * (batch_num - step)
            eta_string = 'eta: {}'.format(str(datetime.timedelta(seconds=int(eta_seconds))))

            time_str = '\t'.join([eta_string, str(time_metrics)])
        else:
            time_str = ''

        msg = log_msg.format(epoch=epoch, epoch_num=epoch_num,
                             step=step, batch_num=batch_num,
                             time_str=time_str, meters=str(self.metrics),
                             memory='max mem: {0:.2f}'.format(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
                             )
        self.logger.info(msg)
        print(msg)
        return

    def epoch_logging(self, epoch, epoch_num, time_metrics=None):
        """
        Print loggings after training of each epoch
        @param epoch:
        @param epoch_num:
        @param time_metrics:
        @return:
        """
        self.logger.info('Epoch: [{epoch}/{epoch_num}] training finished'.format(epoch=epoch, epoch_num=epoch_num))
        log_msg = '\t'.join(['TRN INF:', '{meters}\t'])
        msg = log_msg.format(meters=str(self.metrics))
        if time_metrics is not None:
            msg += 'time: {time:.4f}'.format(time=time_metrics.meters['iter_time'].total)

        self.record_result(epoch)
        self.logger.info(msg)
        print(msg)
        return

    def val_logging(self, epoch):
        msg = '\t'.join(['VAL INF:', '{meters}']).format(meters=self.metrics)
        self.record_result(epoch, 'test')
        return msg


def build_model(args):
    """Import the module "model/[model_name]_model.py"."""
    model = None
    if args.model_type == 'dnn':
        model_file_name = "models." + args.model_type
        modules = importlib.import_module(model_file_name)
        model = modules.__dict__['DNN']
    else:
        model_file_name = "models." + "net"
        modules = importlib.import_module(model_file_name)
        for name, cls in modules.__dict__.items():
            if name.lower() in args.net.lower():
                model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            model_file_name, args.net))
        exit(0)
    else:
        return to_device(args.devices[0], model(args))[0]
