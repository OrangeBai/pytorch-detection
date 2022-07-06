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
        else:
            n = self.attacks['FGSM'].attack(images, labels)

        output_r, output_n = self.dual_net(images, n)
        loss = self.loss_function(output_n, labels)

        # if self.args.float_loss != 0:
        #     float_r = self.dual_net.masked_forward(images)
        #     float_n = self.dual_net.masked_forward(n)
        #     loss += self.args.float_loss * self.set_float_loss(float_r, float_n, labels)

        self.step(loss)

        top1, top5 = accuracy(output_r, labels)
        self.update_metric(top1=(top1, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1),
                           )

    @staticmethod
    def set_float_loss(output_reg, output_noise, labels):
        loss_float = (output_reg - output_noise)
        loss_float = (1 - one_hot(labels, num_classes=loss_float.shape[1])).multiply(loss_float.abs())
        loss_float = loss_float.norm(p=2).mean()
        return loss_float

    def set_lip_loss(self, images, labels):
        perturbation = torch.randn_like(images)
        perturbation = perturbation / 10000
        p_norm = perturbation.norm(p=2, dim=(1, 2, 3)).view(len(perturbation), 1)
        output = self.model(images)
        # if self.args.ord == 'l2':
        #     p_norm = perturbation.norm(p=2, dim=(1, 2, 3)).view(len(perturbation), 1)
        # else:
        #     p_norm = perturbation.norm(p=float('inf'), dim=(1, 2, 3)).view(len(perturbation), 1)
        local_lip = (self.model(images + perturbation) - output) / p_norm
        worst_lip = (1 - one_hot(labels, num_classes=local_lip.shape[1])).multiply(local_lip.abs())
        loss_lip = self.loss_function(output + 2 * self.args.eps * worst_lip, labels)
        return loss_lip
