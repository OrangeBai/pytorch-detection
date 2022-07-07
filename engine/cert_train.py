from torch.nn.functional import one_hot

from core.lip import *
from engine import *
from models.blocks import DualNet


class CertTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.dual_net = DualNet(self.model, args)

    def cert_train_epoch(self, epoch):
        for step in range(self.args.epoch_step):
            images, labels = next(self.inf_loader)
            self.cert_train_step(images, labels)
            if step % self.args.print_every == 0 and step != 0:
                self.step_logging(step, self.args.epoch_step, epoch, self.args.num_epoch, self.inf_loader.metric)

        self.train_logging(epoch, self.args.num_epoch, time_metrics=self.inf_loader.metric)

    def cert_train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)
        if self.args.noise_type == 'noise':
            n = images + torch.randn_like(images, device='cuda') * self.args.noise_sigma
        elif self.args.noise_type == 'FGSM':
            n = self.attacks['FGSM'].attack(images, labels)
        else:
            n = self.attacks['FFGSM'].attack(images, labels)

        output_r, output_n = self.dual_net(images, n)
        if self.args.lip == 1:
            loss = self.loss_function(output_n + output_r / 2, labels)
        else:
            loss = self.loss_function(output_n)

        self.step(loss)

        top1, top5 = accuracy(output_r, labels)
        self.update_metric(top1=(top1, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1),
                           )

