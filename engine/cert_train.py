from torch.nn.functional import one_hot

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
            self.cert_train_step(images, labels)
            if step % self.args.print_every == 0:
                self.step_logging(step, self.args.epoch_step, epoch, self.args.num_epoch, self.inf_loader.metric)

        self.train_logging(epoch, self.args.num_epoch, time_metrics=self.inf_loader.metric)
        self.inf_loader.reset()

    def cert_train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)
        # n = images + torch.sign(torch.randn_like(images, device='cuda')) * 8/255
        n = images + torch.randn_like(images, device='cuda') * 0.1
        # n = self.attacks['FGSM'].attack(images, labels)
        # outputs = self.model(images)
        outputs = self.dual_net.compute_float(images, n)

        # output_reg = self.model(images)
        output_reg = self.dual_net.over_fitting_forward(images)
        # output_reg = self.dual_net.masked_forward(n, 1, 2)

        # output_reg = self.dual_net.masked_forward(images, 1 - 0.00 * (1 - self.trained_ratio), 1)
        # output_flt = self.dual_net.masked_forward(images, 1, 1 + 1 * (1 - self.trained_ratio))
        # noise_output = self.dual_net.masked_forward(n, 1, 1 + 1 * (1 - self.trained_ratio))

        loss_normal = self.loss_function(output_reg, labels)
        # loss_flt = self.loss_function(float_output, labels)
        # loss_float = (output_flt - noise_output)
        # loss_float = (1 - one_hot(labels, num_classes=loss_float.shape[1])).multiply(loss_float.abs())
        # loss_float = self.loss_function(loss_float, labels)
        # loss_float = loss_float.norm(p=2).mean()

        # perturbation = self.lip.attack(images, labels)
        # perturbation = torch.sign(torch.randn_like(images)) * self.args.eps * 2
        # local_lip = (self.model(images + perturbation) - outputs) * 10000
        # worst_lip = (1 - one_hot(labels, num_classes=local_lip.shape[1])).multiply(local_lip.abs())
        # loss_lip = self.loss_function(outputs + worst_lip * self.trained_ratio, labels)

        # loss = loss_normal
        loss = loss_normal
        self.step(loss)

        top1, top5 = accuracy(outputs, labels)
        self.update_metric(top1=(top1, len(images)), top5=(top5, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1),
                           mask=(self.flt_net.mask_ratio, 1),
                           # l1_lip=(local_lip.norm(p=float('inf'), dim=1).mean(), len(images)),
                           # l2_lip=(local_lip.norm(p=2, dim=1).mean(), len(images))
                           )
        # if self.est_lip % self.args.fre_est_lip == 0:
        #     self.update_metric(ratio=(ratio.mean(), len(images)))
        # self.est_lip += 1
