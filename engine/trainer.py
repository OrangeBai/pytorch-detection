from engine.cert_train import *
from engine.adv_train import *
from engine.gen_trainer import *
from engine.prune_train import *


class Trainer(AdvTrainer, CertTrainer, GenTrainer, PruTrainer):
    def __init__(self, args):
        super().__init__(args)

    def train_model(self):
        self.warmup()
        best_acc = 0
        # self.validate_epoch(-1)
        for epoch in range(self.args.num_epoch):
            self.train_epoch(epoch)
            acc = self.validate_epoch(epoch)
            if acc > best_acc:
                best_acc = acc
                self.model.save_model(self.args.model_dir, 'cur_best')
            self.record_result(epoch)
            self.model.train()
        self.std_validate_epoch(-1)
        self.model.save_model(self.args.model_dir)
        self.save_result(self.args.model_dir)

    def train_epoch(self, epoch):
        self.inf_loader.reset()
        if self.args.train_mode == 'cer':
            self.cert_train_epoch(epoch)
        elif self.args.train_mode == 'std':
            self.std_train_epoch(epoch)
        elif self.args.train_mode == 'adv':
            self.adv_train_epoch(epoch)
        elif self.args.train_mode == 'gen':
            self.gen_train_epoch(epoch)
        elif self.args.train_mode == 'pru':
            self.std_train_epoch(epoch)
        else:
            raise NameError

    def validate_epoch(self, epoch):
        if self.args.val_mode in ['adv', 'cer']:
            return self.adv_validate_epoch(epoch)
        elif self.args.val_mode == 'std':
            return self.std_validate_epoch(epoch)
        elif self.args.val_mode == 'gen':
            if epoch % self.args.prune_every == 0:
                return self.gen_validate_epoch(epoch)
            else:
                return self.std_validate_epoch(epoch)
        elif self.args.val_mode == 'pru':
            if epoch % self.args.prune_every == 0:
                return self.prune_validate_epoch(epoch)
            else:
                return self.std_validate_epoch(epoch)
        else:
            raise NameError

    def train_step(self, images, labels):
        if self.args.train_mode == 'cer':
            self.cert_train_step(images, labels)
        elif self.args.train_mode == 'std':
            self.std_train_step(images, labels)
        elif self.args.train_mode == 'adv':
            self.adv_train_step(images, labels)
        elif self.args.train_mode == 'prune':
            self.std_train_step(images, labels)
        else:
            raise NameError
