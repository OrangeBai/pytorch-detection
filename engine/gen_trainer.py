import time

from core.lip import *
from core.prune import *
from engine import BaseTrainer
from models.blocks import DualNet


class GenTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.dual_net = DualNet(self.model, args)

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
        self.gen_weight(net_same_all)

        acc = self.metrics.top1.global_avg
        msg = self.val_logging(epoch) + '\ttime:{0:.4f}'.format(time.time() - start)
        self.logger.info(msg)
        print(msg)

        self.model.train()
        return acc

    def gen_weight(self, net_same_all):
        block_idx = 0
        for m in self.model.layers.children():
            if m in [LinearBlock, ConvBlock]:
                self.reset_weight(m, net_same_all[block_idx])
                block_idx += 1
        return

    def reset_weight(self, m, block_same):
        if type(m) == ConvBlock:
            n = m.Conv.kernel_size[0] * m.Conv.kernel_size[1] * m.Conv.out_channels
            dead_ids = block_same[0] > len(self.test_loader.dataset) * 0.98
            m.Conv.weight[dead_ids] = m.Conv.weight[dead_ids].data.normal_(0, math.sqrt(2. / n))
        elif type(m) == LinearBlock:
            dead_ids = block_same[0] > len(self.test_loader.dataset) * 0.98
            m.FC.weight[dead_ids] = m.FC.weight[dead_ids].data_normal_(0, math.sqrt(2 / m.FC.out_channels))
        return
