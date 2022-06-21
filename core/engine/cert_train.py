from core.engine.trainer import *
from Lip.utils import estimate_lip

class CertTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.attack_args = {
            'mean': self.mean,
            'std': self.std,
            'eps': self.args.eps,
            'alpha': self.args.alpha,
            'ord': self.args.ord
        }
        self.num_flt_est = args.num_flt_est
        self.lip = set_attack(self.model, 'Lip', self.args.devices[0], **self.attack_args)
        self.est_lip = False

    @property
    def set_attack(self):
        return {'FGSM': set_attack(self.model, 'FGSM', self.args.devices[0], **self.attack_args),
                'PGD': set_attack(self.model, 'PGD', self.args.devices[0], **self.attack_args),
                # 'CW': set_attack(self.model, 'CW', self.args.devices[0], **self.attack_args)
                }

    def train_epoch(self, epoch, *args, **kwargs):

        for step in range(self.args.epoch_step):
            images, labels = next(self.inf_loader)
            if step % self.args.fre_lip_est == 0:
                self.est_lip = True
            self.train_step(images, labels)
            self.est_lip = False
            if step % self.args.print_every == 0:
                self.step_logging(step, self.args.epoch_step, epoch, self.args.num_epoch, self.inf_loader.metric)

        self.train_logging(epoch, self.args.num_epoch, time_metrics=self.inf_loader.metric)
        self.inf_loader.reset()

    def train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)
        self.optimizer.zero_grad()
        # if self.est_lip:
        #     t = time.time()
        #     ratio = estimate_lip(self.args, self.model, images, self.num_flt_est)
        #     print(t - time.time())
        #     ratio = torch.tensor(ratio).view(len(ratio), 1).cuda()
        # else:
        #     ratio = self.metrics.ratio.avg

        perturbation = self.lip.attack(images, labels)
        outputs = self.model(images)
        local_lip = (self.model(images + perturbation) - outputs) * 10000 * 0.86
        local_lip = (1 - one_hot(labels, num_classes=self.args.num_cls)).mul(local_lip).abs()

        loss_nor = self.loss_function(outputs, labels)
        loss_reg = self.loss_function(outputs + 5 * local_lip, labels)
        loss = self.trained_ratio * loss_reg + (1 - self.trained_ratio) * loss_nor
        loss.backward()
        self.step()

        top1, top5 = accuracy(outputs, labels)
        self.update_metric(top1=(top1, len(images)), top5=(top5, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1),
                           l1_lip=(local_lip.norm(p=1, dim=1).mean(), len(images)),
                           l2_lip=(local_lip.norm(p=2, dim=1).mean(), len(images)))
        # if self.est_lip:
        #     self.update_metric(ratio=(ratio.mean(), len(images)))

    def validate_epoch(self, epoch):
        start = time.time()
        self.model.eval()
        for images, labels in self.test_loader:
            images, labels = to_device(self.args.devices[0], images, labels)
            pred = self.model(images)
            top1, top5 = accuracy(pred, labels)
            self.update_metric(top1=(top1, len(images)), top5=(top5, len(images)))
            for name, attack in self.set_attack.items():
                adv = attack.attack(images, labels)
                pred_adv = self.model(adv)
                top1, top5 = accuracy(pred_adv, labels)
                update_times = {name + 'top1': (top1, len(images)),
                                name + 'top5': (top5, len(images))}
                self.update_metric(**update_times)
        self.model.train()
        msg = self.val_logging(epoch) + '\ttime:{0:.4f}'.format(time.time() - start)

        self.logger.info(msg)
        print(msg)
        return

    def train_model(self):
        self.warmup()
        for epoch in range(self.args.num_epoch):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            self.record_result(epoch)

        self.model.save_model(self.args.model_dir)
        self.model.save_result(self.args.model_dir)
