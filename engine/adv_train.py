from engine.base_trainer import *
from convex_adversarial import robust_loss

class AdvTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.attack_args = {
            'mean': self.mean,
            'std': self.std,
            'eps': self.args.eps,
            'alpha': self.args.alpha,
            'ord': self.args.ord
        }
        self.attacks = self.set_attack

    @property
    def set_attack(self):
        return {'FGSM': set_attack(self.model, 'FGSM', self.args.devices[0], **self.attack_args),
                'PGD': set_attack(self.model, 'PGD', self.args.devices[0], **self.attack_args),
                # 'CW': set_attack(self.model, 'CW', self.args.devices[0], **self.attack_args)
                }

    def adv_train_epoch(self, epoch, *args, **kwargs):

        for step in range(self.args.epoch_step):
            images, labels = next(self.inf_loader)
            self.adv_train_step(images, labels)
            if step % self.args.print_every == 0:
                self.step_logging(step, self.args.epoch_step, epoch, self.args.num_epoch, self.inf_loader.metric)

        self.train_logging(epoch, self.args.num_epoch, time_metrics=self.inf_loader.metric)
        self.inf_loader.reset()

    def adv_train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)
        self.optimizer.zero_grad()
        adv_images = self.attacks[self.args.attack].attack(images, labels)
        outputs = self.model(adv_images)
        loss = self.loss_function(outputs, labels)
        loss.backward()
        self.step()
        local_lip = self.lip.attack(images, outputs)
        top1, top5 = accuracy(outputs, labels)
        self.update_metric(top1=(top1, len(images)), top5=(top5, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1),
                           l1_lip=(local_lip.norm(p=1, dim=1).mean(), len(images)),
                           l2_lip=(local_lip.norm(p=2, dim=1).mean(), len(images)))

    def adv_validate_epoch(self, epoch):
        start = time.time()
        self.model.eval()
        for images, labels in self.test_loader:
            images, labels = to_device(self.args.devices[0], images, labels)
            pred = self.model(images)
            top1, top5 = accuracy(pred, labels)
            self.update_metric(top1=(top1, len(images)), top5=(top5, len(images)))
            for name, attack in self.attacks.items():
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
