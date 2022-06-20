from core.engine.trainer import *


class CertTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        attack_args = {
            'mean': self.mean,
            'std': self.std,

        }
        self.fgsm = set_attack(self.model, 'FGSM', self.args.devices[0], mean=self.mean, std=self.std, *args, **kwargs)
        self.pgd = set_attack(self.model, 'PGD', self.args.devices[0], mean=mean, std=std, *args, **kwargs)

    def train_epoch(self, cur_epoch, *args, **kwargs):
        lip_mean = []
        for cur_step in range(self.args.epoch_step):
            images, labels = next(self.inf_loader)
            self.train_step(images, labels)
            images, labels = to_device(self.args.devices[0], images, labels)
            self.optimizer.zero_grad()

            lip = LipAttack(model.model, model.args.devices[0], eps=1 / 255, mean=mean, std=std)

            # if cur_step % 10 == 0 or cur_step == 0:
            # lip_ratio = torch.tensor(estimate_lip(model, images, 16)).view(len(images), 1).repeat(1, 10).cuda()
            # lip_mean += [lip_ratio.cpu()]
            # else:
            # lip_ratio = torch.stack(lip_mean).mean()
            perturbation = lip.attack(images, labels)
            # r = torch.randn(images.shape).cuda()
            # perturbation = images + r / r.norm(p=2, dim=(1, 2, 3)).view(len(r), 1, 1, 1) * 0.01
            outputs = self.model(images)
            certified_res = self.model(images + perturbation) - outputs
            c2 = certified_res.abs().max(axis=1)[0].view(len(certified_res), 1).repeat(1, 10)
            local_lip = (1 - one_hot(labels, num_classes=self.args.num_cls)).mul(c2).abs() * 1000 * 0.86
            if model.trained_ratio() < 0.3:
                rate = 1 / 4
            elif model.trained_ratio() < 0.6:
                rate = 2 / 4
            elif model.trained_ratio() < 0.8:
                rate = 3 / 4
            else:
                rate = 1
            loss_nor = model.loss_function(outputs, labels)
            loss_reg = model.loss_function(outputs + rate * 15 * local_lip.detach(), labels) \
                       + 0.1 * local_lip.norm(p=float('inf'), dim=1).mean()
            loss = loss_reg
            loss.backward()
            model.step()

            top1, top5 = accuracy(outputs, labels)
            model.metrics.update(top1=(top1, len(images)), top5=(top5, len(images)), loss=(loss, len(images)),
                                 lr=(model.optimizer.param_groups[0]['lr'], 1), aa=(local_lip.mean(), 1))
            model.metrics.synchronize_between_processes()
            if cur_step % args.print_every == 0:
                self.step_logging(cur_step, args.epoch_step, cur_epoch, args.num_epoch, inf_loader.metric)

        self.train_logging(cur_epoch, args.num_epoch, time_metrics=inf_loader.metric)
        self.inf_loader.reset()