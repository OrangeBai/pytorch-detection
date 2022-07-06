from engine.cert_train import CertTrainer
from engine.adv_train import AdvTrainer
from engine.gen_trainer import GenTrainer


class Trainer(AdvTrainer, CertTrainer, GenTrainer):
    def __init__(self, args):
        super().__init__(args)

    def train_model(self):
        self.warmup()
        best_acc = 0
        for epoch in range(self.args.num_epoch):
            self.train_epoch(epoch)
            acc = self.validate_epoch(epoch)
            if acc > best_acc:
                best_acc = acc
                self.model.save_model(self.args.model_dir, 'cur_best')
            self.record_result(epoch)
            self.model.train()
        self.normal_validate_epoch(-1)
        self.model.save_model(self.args.model_dir)
        self.save_result(self.args.model_dir)

    def train_epoch(self, epoch):
        self.inf_loader.reset()
        if self.args.train_mode == 'cer':
            self.cert_train_epoch(epoch)
        elif self.args.train_mode == 'std':
            self.normal_train_epoch(epoch)
        elif self.args.train_mode == 'adv':
            self.adv_train_epoch(epoch)
        elif self.args.train_mode == 'gen':
            self.gen_train_epoch(epoch)
        # elif self.args.train_mode == 'prune':
        #     train_epoch = self.prune_train_epoch
        else:
            raise NameError

    def validate_epoch(self, epoch):
        if self.args.val_mode in ['adv', 'cer']:
            return self.adv_validate_epoch(epoch)
        elif self.args.val_mode == 'std':
            return self.normal_validate_epoch(epoch)
        # elif self.args.val_mode == 'prune':
        #     validate_epoch = self.prune_validate_epoch
        else:
            raise NameError

    def train_step(self, images, labels):
        if self.args.train_mode == 'cer':
            self.cert_train_step(images, labels)
        elif self.args.train_mode == 'std':
            self.normal_train_step(images, labels)
        elif self.args.train_mode == 'adv':
            self.adv_train_step(images, labels)
        # elif self.args.train_mode == 'prune':
        #     train_epoch = self.prune_train_epoch
        else:
            raise NameError
