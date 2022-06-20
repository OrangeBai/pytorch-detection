from core.engine.trainer import *


class CertTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        attack_args = {
            'mean': self.mean,
            'std': self.std,
            'eps': args.eps,
            'alpha': args.alpha
        }
        self.num_flt_est = args.num_flt_est

        self.fgsm = set_attack(self.model, 'FGSM', self.args.devices[0],  **attack_args)
        self.pgd = set_attack(self.model, 'PGD', self.args.devices[0], **attack_args)
        self.lip = set_attack(self.model, 'Lip', self.args.devices[0],  **attack_args)
        self.lip_metric = SmoothedValue()

    def train_epoch(self, cur_epoch, *args, **kwargs):

        for cur_step in range(self.args.epoch_step):
            images, labels = next(self.inf_loader)
            self.train_step(images, labels)



            model.step()

            top1, top5 = accuracy(outputs, labels)
            model.metrics.update(top1=(top1, len(images)), top5=(top5, len(images)), loss=(loss, len(images)),
                                 lr=(model.optimizer.param_groups[0]['lr'], 1), aa=(local_lip.mean(), 1))
            model.metrics.synchronize_between_processes()
            if cur_step % args.print_every == 0:
                self.step_logging(cur_step, args.epoch_step, cur_epoch, args.num_epoch, inf_loader.metric)

        self.train_logging(cur_epoch, args.num_epoch, time_metrics=inf_loader.metric)
        self.inf_loader.reset()

    def train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)
        self.optimizer.zero_grad()
        if self.est_lip:
            ratio = estimate_lip(self.model, images, self.num_flt_est)
        else:
            ratio = self.lip_metric.avg

        perturbation = self.lip.attack(images, labels)
        outputs = self.model(images)
        certified_res = self.model(images + perturbation) - outputs
        local_lip = (1 - one_hot(labels, num_classes=self.args.num_cls)).mul(certified_res).abs() * 1000 * 0.86

        loss_nor = self.loss_function(outputs, labels)
        # if self.trained_ratio() < 0.3:
        #     rate = 1 / 4
        # elif self.trained_ratio() < 0.6:
        #     rate = 2 / 4
        # elif self.trained_ratio() < 0.8:
        #     rate = 3 / 4
        # else:
        #     rate = 1
        loss_reg = self.loss_function(outputs + rate * ratio * local_lip.detach(), labels) \
                   + 0.1 * local_lip.norm(p=float('inf'), dim=1).mean()
        loss = loss_reg + loss_nor
        self.step(loss)

        top1, top5 = accuracy(outputs, labels)
        self.update_metric(top1=(top1, len(images)), top5=(top5, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1))