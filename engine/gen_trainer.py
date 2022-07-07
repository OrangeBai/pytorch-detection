import time

import torch

from core.prune import *
from engine import BaseTrainer
from models.blocks import DualNet


class GenTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.dual_net = DualNet(self.model, args)
        self.conv_dn_rate = args.conv_dn_rate
        self.linear_dn_rate = args.linear_dn_rate

    def gen_train_epoch(self, epoch):
        for step in range(self.args.epoch_step):
            images, labels = next(self.inf_loader)
            self.gen_train_step(images, labels)
            if step % self.args.print_every == 0 and step != 0:
                self.step_logging(step, self.args.epoch_step, epoch, self.args.num_epoch, self.inf_loader.metric)

        self.train_logging(epoch, self.args.num_epoch, time_metrics=self.inf_loader.metric)

    def gen_train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)

        output = self.dual_net.dn_forward(images)
        loss = self.loss_function(output, labels)
        self.step(loss)

        top1, top5 = accuracy(output, labels)
        self.update_metric(top1=(top1, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1),
                           )

    def gen_validate_epoch(self, epoch):
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
        ap_hook.remove()
        num_dn = self.gen_weight(net_same_all)
        acc = self.metrics.top1.global_avg
        msg = self.val_logging(epoch) + '\ttime:{0:.4f}'.format(time.time() - start)
        msg += '\tGen validation finished, found {0} dead neurons'.format(num_dn)
        self.logger.info(msg)
        print(msg)

        self.model.train()
        return acc

    def gen_weight(self, net_same_all):
        block_idx = 0
        num_dn = 0
        for m in self.model.layers.children():
            if type(m) in [LinearBlock, ConvBlock]:
                num_dn += self.reset_weight(m, net_same_all[block_idx])
                block_idx += 1
        return num_dn

    def reset_weight(self, m, block_same):
        if type(m) == ConvBlock:
            dead_ids = block_same[0] > len(self.test_loader.dataset) * self.args.conv_dn_rate
            weight = m.Conv.weight.data
            new_weight = nn.init.xavier_uniform_(torch.empty_like(weight))
            weight[dead_ids] = new_weight[dead_ids]
        elif type(m) == LinearBlock:
            dead_ids = block_same[0] > len(self.test_loader.dataset) * self.args.linear_dn_rate
            weight = m.FC.weight.data
            new_weight = nn.init.xavier_uniform_(torch.empty_like(weight))
            weight[dead_ids] = new_weight[dead_ids]
        else:
            raise NotImplementedError
        return dead_ids.sum()
