from torch.nn.functional import one_hot
from core.pattern import *
from core.lip import *
from core.utils import *
from engine import *
from models.blocks import FloatNet, DualNet


class CertTrainer(AdvTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.num_flt_est = args.num_flt_est
        self.est_lip = 0
        self.flt_net = FloatNet(self.model)
        self.dual_net = DualNet(self.model, set_gamma(args.activation))
        # self.attacks = self.set_attack

    def cert_train_epoch(self, epoch):
        self.est_lip = 0
        for step in range(self.args.epoch_step):
            images, labels = next(self.inf_loader)
            self.normal_train_step(images, labels)
            if step % self.args.print_every == 0:
                self.step_logging(step, self.args.epoch_step, epoch, self.args.num_epoch, self.inf_loader.metric)

        self.train_logging(epoch, self.args.num_epoch, time_metrics=self.inf_loader.metric)
        self.inf_loader.reset()

    def pruning_val(self, epoch, test_loader):
        self.model.eval()
        ap_hook = ModelHook(self.model, retrieve_input_hook)
        metric = MetricLogger()
        storage = []
        net_same_all = []
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = to_device(self.args.devices[0], images, labels)
            pre_ori = self.model(images)
            top1, top5 = accuracy(pre_ori, labels)
            unpacked = ap_hook.retrieve_res(unpack2)
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