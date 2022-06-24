from torch.nn.functional import one_hot
from core.pattern import *
from core.lip import *
from engine import *
from core.utils import *


class CertTrainer(AdvTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.num_flt_est = args.num_flt_est
        self.est_lip = False

    def cert_train_epoch(self, epoch):

        for step in range(self.args.epoch_step):
            images, labels = next(self.inf_loader)
            if step % self.args.fre_lip_est == 0:
                self.est_lip = True
            self.cert_train_step(images, labels)
            self.est_lip = False
            if step % self.args.print_every == 0:
                self.step_logging(step, self.args.epoch_step, epoch, self.args.num_epoch, self.inf_loader.metric)

        self.train_logging(epoch, self.args.num_epoch, time_metrics=self.inf_loader.metric)
        self.inf_loader.reset()

    def cert_train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)
        # if self.est_lip:
        #     ratio = estimate_lip(self.args, self.model, images, self.num_flt_est)
        #     ratio = torch.tensor(ratio).view(len(ratio), 1).cuda()
        # else:
        #     ratio = self.metrics.ratio.avg

        perturbation = self.lip.attack(images, labels)

        outputs = self.model(images)
        local_lip = (self.model(images + perturbation) - outputs) * 10000
        if self.args.ord == 'l2':
            worst_lip = (1 - one_hot(labels, num_classes=self.args.num_cls)).mul(local_lip).abs() * self.args.eps * 4
        else:
            eps = torch.norm(torch.ones(images[0].shape) * self.args.eps, p=2)
            worst_lip = (1 - one_hot(labels, num_classes=self.args.num_cls)).mul(local_lip).abs() * eps * 4

        self.optimizer.zero_grad()
        loss_nor = self.loss_function(outputs, labels)
        loss_reg = self.loss_function(outputs + 2 * worst_lip, labels)
        loss = self.trained_ratio * loss_reg + (1 - self.trained_ratio) * loss_nor
        loss.backward()
        self.step()

        top1, top5 = accuracy(outputs, labels)
        self.update_metric(top1=(top1, len(images)), top5=(top5, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1),
                           l1_lip=(local_lip.norm(p=float('inf'), dim=1).mean(), len(images)),
                           l2_lip=(local_lip.norm(p=2, dim=1).mean(), len(images)))
        # if self.est_lip:
        #     self.update_metric(ratio=(ratio.mean(), len(images)))

