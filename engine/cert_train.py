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
        self.inf_loader.reset()

    def cert_train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)
        if self.args.noise_type == 'noise':
            n = images + torch.randn_like(images, device='cuda') * self.args.noise_sigma
        else:
            n = self.attacks['FGSM'].attack(images, labels)

        if self.args.eta_fixed != 0 or self.args.eta_float != 0:
            output_reg, output_noise = self.dual_net(images, n)
        elif self.args.eta_dn != 0:
            output_reg = self.dual_net.dn_forward(images)
            output_noise = None
        else:
            if self.args.cert_input != 'noise':
                raise ArithmeticError('Not using AP training, set train_mode to normal')
            output_reg = self.model(n)
            output_noise = None

        loss = self.set_loss(images, labels, output_reg, output_noise)
        self.step(loss)

        top1, top5 = accuracy(output_reg, labels)
        self.update_metric(top1=(top1, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1),
                           )

    def set_loss_default(self, output_reg, output_noise, labels):
        if self.args.cert_input == 'noise':
            loss_normal = self.loss_function(output_noise, labels)
        else:
            loss_normal = self.loss_function(output_reg, labels)
        return loss_normal

    def set_float_loss(self, output_reg, output_noise, labels):
        if self.args.float_loss != 0:
            loss_float = (output_reg - output_noise)
            loss_float = (1 - one_hot(labels, num_classes=loss_float.shape[1])).multiply(loss_float.abs())
            loss_float = loss_float.norm(p=2).mean()
            return loss_float
        else:
            return 0

    def set_lip_loss(self, images, output_reg, labels):
        perturbation = self.lip.attack(images, labels)
        local_lip = (self.model(images + perturbation) - output_reg) * 10000
        worst_lip = (1 - one_hot(labels, num_classes=local_lip.shape[1])).multiply(local_lip.abs())
        loss_lip = self.loss_function(output_reg + worst_lip * self.args.eps, labels)
        return loss_lip

    def set_loss(self, images, labels, output_reg, output_noise=None):
        if output_noise is None:
            loss_normal = self.loss_function(output_reg, labels)
            float_loss = 0
        else:
            loss_normal = self.set_loss_default(output_reg, output_noise, labels)
            float_loss = self.set_float_loss(output_reg, output_noise, labels)
        if self.args.lip:
            loss_lip = self.set_lip_loss(images, output_reg, labels) * self.trained_ratio
            loss_normal = loss_normal * (1 - self.trained_ratio) + loss_lip * self.trained_ratio

        return loss_normal + self.args.float_loss * float_loss
