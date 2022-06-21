from core.engine.trainer import *
from Lip.utils import *


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

        self.fgsm = set_attack(self.model, 'FGSM', self.args.devices[0], **attack_args)
        self.pgd = set_attack(self.model, 'PGD', self.args.devices[0], **attack_args)
        self.lip = set_attack(self.model, 'Lip', self.args.devices[0], **attack_args)
        self.lip_metric = SmoothedValue()
        self.est_lip = False

    def train_epoch(self, epoch, *args, **kwargs):

        for step in range(self.args.epoch_step):
            images, labels = next(self.inf_loader)
            if step % 20 == 0:
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
        if self.est_lip:
            t = time.time()
            ratio = estimate_lip(self.args, self.model, images, self.num_flt_est)
            print(t-time.time())
            self.lip_metric.update(ratio, len(images))
            ratio = torch.tensor(ratio).view(len(ratio), 1).cuda()
        else:
            ratio = self.lip_metric.avg

        perturbation = self.lip.attack(images, labels)
        outputs = self.model(images)
        local_lip = (self.model(images + perturbation) - outputs) * 10000 * 0.86
        local_lip = (1 - one_hot(labels, num_classes=self.args.num_cls)).mul(local_lip).abs()

        loss_nor = self.loss_function(outputs, labels)
        loss_reg = self.loss_function(outputs + ratio * local_lip, labels)
        loss = self.trained_ratio * loss_reg + loss_nor * (1 - self.trained_ratio)
        loss.backward()
        self.step()

        top1, top5 = accuracy(outputs, labels)
        self.update_metric(top1=(top1, len(images)), top5=(top5, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1),
                           l1_lip=(local_lip.norm(p=1), len(images)), l2_lip=(local_lip.norm(p=2), len(images)))

    def train_model(self):
        self.warmup()

        for epoch in range(self.args.num_epoch):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)

        self.model.save_model(self.args.model_dir)
        self.model.save_result(self.args.model_dir)
