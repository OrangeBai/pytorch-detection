from torch.nn.functional import one_hot

from core.lip import *
from engine import *
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
