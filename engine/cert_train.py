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
        elif self.args.nosie_type == 'images':
            n = images
        else:
            raise NameError

        if self.args.ord == 'l2':
            noise = torch.randn_like(images)
            noise_norm = noise.view(len(noise), -1).norm(p=2, dim=-1) * self.args.noise_eps
            x_2 = (images + noise / noise_norm).detach()
        else:
            noise = torch.sign(torch.randn_like(images)) * self.args.noise_eps
            x_2 = (images + noise).detach()

        if self.args.eta_fixed != 0 or self.args.eta_float != 0:
            eta_fixed = self.args.eta_fixed * (1 - self.trained_ratio)
            eta_float = self.args.eta_float * (1 - self.trained_ratio)
            output, output_n, df = self.dual_net(n, x_2, eta_fixed, eta_float)
            loss = self.loss_function(output, labels)
            if self.args.lip_loss != 0:
                loss += torch.log(df) * self.args.lip_loss
        else:
            output = self.model(n)
            loss = self.loss_function(output, labels)

        self.step(loss)
        top1, top5 = accuracy(output, labels)
        self.update_metric(top1=(top1, len(images)), loss=(loss, len(images)),
                           lr=(self.get_lr(), 1), mask=(self.dual_net.mask_ratio, len(images)))

    @staticmethod
    def set_float_loss(output_reg, output_noise, labels):
        loss_float = (output_reg - output_noise)
        loss_float = (1 - one_hot(labels, num_classes=output_noise.shape[1])).multiply(loss_float.abs())
        loss_float = loss_float.norm(p=2).mean()
        return loss_float
