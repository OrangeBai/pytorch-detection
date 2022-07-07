from core.prune import *
from models.blocks import *
from models.base_model import build_model
from core.lip import *
from core.utils import *
from engine.base_trainer import BaseTrainer
import time


class PruTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.conv_prune_rate = args.conv_prune_rate
        self.linear_prune_rate = args.linear_prune_rate

    def prune_validate_epoch(self, epoch):
        self.model.eval()
        start = time.time()
        ap_hook = ModelHook(self.model, set_input_hook)
        net_same_all = None
        for idx, (images, labels) in enumerate(self.test_loader):
            images, labels = to_device(self.args.devices[0], images, labels)
            pre_ori = self.model(images)
            top1, top5 = accuracy(pre_ori, labels)
            self.update_metric(top_1=(top1, self.args.batch_size), top_5=(top5, self.args.batch_size))

            unpacked = ap_hook.retrieve_res(unpack)
            net_same_all = compute_mean(net_same_all, find_dead_neuron(unpacked, [0]))


        acc = self.metrics.top1.global_avg

        net_same_all.insert(0, [])
        self.model = self.model.cpu()
        block_counter = 1
        new_model = []
        new_size = []
        conv_prune_num = len(self.test_loader.dataset) * self.conv_prune_rate
        linear_prune_num = len(self.test_loader.dataset) * self.linear_prune_rate
        for m in self.model.layers.children():
            if type(m) in [LinearBlock, ConvBlock, BasicBlock, BottleNeck]:
                cur_block_ps = net_same_all[block_counter]
                pre_block_ps = net_same_all[block_counter - 1]
                m, shape = prune_block(m, cur_block_ps, pre_block_ps, conv_prune_num, linear_prune_num)
                new_model += [m]
                new_size += [shape]
                block_counter += 1
            else:
                new_model += [m]
                if type(m) == nn.MaxPool2d:
                    new_size += ['M']
        self.model.layers = nn.Sequential(*new_model)
        self.args.config = new_size

        self.model = build_model(self.args)
        wt = self.model.state_dict()
        self.model.load_weights(wt)

        self.model = self.model.cuda()
        self.model.train()
        self.optimizer = init_optimizer(self.args, self.model)
        last_epoch = self.lr_scheduler.last_epoch
        self.lr_scheduler = init_scheduler(self.args, self.optimizer)
        self.lr_scheduler.last_epoch = last_epoch

        msg = self.val_logging(epoch) + '\ttime:{0:.4f}'.format(time.time() - start)
        self.logger.info(msg)

        print(msg)
        print('Triming to {0}'.format(new_size))
        self.logger.info('Triming to {0}'.format(new_size))
        return acc
