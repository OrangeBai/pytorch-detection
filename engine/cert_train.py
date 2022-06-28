from torch.nn.functional import one_hot
from core.lip import *
from engine import *
from core.utils import *


class CertTrainer(AdvTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.num_flt_est = args.num_flt_est
        self.est_lip = 0

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
        # if self.est_lip % self.args.fre_est_lip == 0:
        #     ratio = estimate_lip(self.args, self.model, images, self.num_flt_est)
        #     ratio = torch.tensor(ratio).view(len(ratio), 1).cuda()
        # else:
        #     ratio = self.metrics.ratio.avg
        pattern_hook = ModelHook(self.model, set_input_hook, device='gpu')
        outputs = self.model(images)
        pt = pattern_hook.retrieve_res(unpack)
        pattern_hook.remove()
        pt = [i for j in pt[-2:] for i in j]
        pt_loss = torch.tensor(1.0).cuda()
        for p in pt:
            pt_loss += p[p.abs() < 1e-1].abs().sum()
        pt_loss = torch.log(pt_loss) * 0.01
        perturbation = self.lip.attack(images, labels)

        local_lip = (self.model(images + perturbation) - outputs) * 10000
        if self.args.ord == 'l2':
            worst_lip = (1 - one_hot(labels, num_classes=self.args.num_cls)).mul(local_lip).abs()
        else:
            # eps = torch.norm(torch.ones(images[0].shape) * self.args.eps, p=2)
            worst_lip = (1 - one_hot(labels, num_classes=self.args.num_cls)).mul(local_lip).abs()
        self.optimizer.zero_grad()
        loss_nor = self.loss_function(outputs, labels)
        loss_reg = self.loss_function(outputs + self.trained_ratio * worst_lip, labels)
        lip_loss = torch.log(1 + worst_lip.norm(p=2, dim=-1).mean()) * 0.01
        loss = loss_nor - pt_loss + loss_reg * self.trained_ratio
        loss.backward()
        self.step()

        top1, top5 = accuracy(outputs, labels)
        self.update_metric(top1=(top1, len(images)), top5=(top5, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1),
                           l1_lip=(local_lip.norm(p=float('inf'), dim=1).mean(), len(images)),
                           l2_lip=(local_lip.norm(p=2, dim=1).mean(), len(images)))
        # if self.est_lip % self.args.fre_est_lip == 0:
        #     self.update_metric(ratio=(ratio.mean(), len(images)))
        # self.est_lip += 1
