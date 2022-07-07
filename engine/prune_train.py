from core.prune import *
from models.blocks import *
from core.lip import *
from core.utils import *
from engine.base_trainer import BaseTrainer


class PruTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.conv_prune_rate = args.conv_prune_rate
        self.linear_prune_rate = args.linear_prune_rate

    def prune_validate_epoch(self, epoch):
        self.model.eval()
        ap_hook = ModelHook(self.model, set_input_hook)
        metric = MetricLogger()
        batch_ps = []
        for idx, (images, labels) in enumerate(self.test_loader):
            images, labels = to_device(self.args.devices[0], images, labels)
            pre_ori = self.model(images)
            top1, top5 = accuracy(pre_ori, labels)
            metric.update(top1_avd=(top1, self.args.batch_size), top5_adv=(top5, self.args.batch_size))

            unpacked = ap_hook.retrieve_res(unpack)
            batch_ps += [find_dead_neuron(unpacked, [0])]

        net_same_all = compute_mean(batch_ps, idx)
        unpacked = ap_hook.retrieve_res(unpack)
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
        self.model.load_weights(wt)

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