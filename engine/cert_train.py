from torch.nn.functional import one_hot

from core.lip import *
from core.utils import *
from engine import *


class CertTrainer(AdvTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.num_flt_est = args.num_flt_est
        self.est_lip = 0
        self.flt_net = FloatNet(self.model)
        # self.attacks = self.set_attack

    def cert_train_epoch(self, epoch):
        self.est_lip = 0
        for step in range(self.args.epoch_step):
            images, labels = next(self.inf_loader)
            self.cert_train_step(images, labels)
            if step % self.args.print_every == 0:
                self.step_logging(step, self.args.epoch_step, epoch, self.args.num_epoch, self.inf_loader.metric)

        self.train_logging(epoch, self.args.num_epoch, time_metrics=self.inf_loader.metric)
        self.inf_loader.reset()

    def cert_train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)
        n = images + torch.randn_like(images, device='cuda') * 0.1
        outputs = self.model(n)
        loss_nor = self.loss_function(outputs, labels)

        flt_outputs = self.flt_net(images, 0.01, 'float', skip=0)
        flt_outputs_n = self.flt_net.mask_forward(n, inverse=False)
        worst_flt = (1-one_hot(labels, num_classes=flt_outputs.shape[1])).multiply(flt_outputs - flt_outputs_n)
        worst_flt = worst_flt.norm(p=2, dim=-1).mean()
        loss = loss_nor + worst_flt * 0.01
        self.step(loss)

        top1, top5 = accuracy(outputs, labels)
        self.update_metric(top1=(top1, len(images)), top5=(top5, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1),
                           mask=(self.flt_net.mask_ratio, 1)
                           # l1_lip=(local_lip.norm(p=float('inf'), dim=1).mean(), len(images)),
                           # l2_lip=(local_lip.norm(p=2, dim=1).mean(), len(images))
                           )
        # if self.est_lip % self.args.fre_est_lip == 0:
        #     self.update_metric(ratio=(ratio.mean(), len(images)))
        # self.est_lip += 1
