from engine.cert_train import CertTrainer
from core.BCP_utils import argparser
from engine.cert_train import CertTrainer


class Trainer(CertTrainer):
    def __init__(self, args):
        super().__init__(args)

    def train_model(self):
        self.warmup()
        for epoch in range(self.args.num_epoch):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            self.record_result(epoch)

        self.model.save_model(self.args.model_dir)
        self.save_result(self.args.model_dir)

    def train_epoch(self, epoch):
        if self.args.train_mode == 'cert':
            self.cert_train_epoch(epoch)
        elif self.args.train_mode == 'normal':
            self.normal_train_epoch(epoch)
        elif self.args.train_mode == 'adv':
            self.adv_train_epoch(epoch)
        # elif self.args.train_mode == 'prune':
        #     train_epoch = self.prune_train_epoch
        else:
            raise NameError

    def validate_epoch(self, epoch):
        if self.args.val_mode in ['normal', 'adv', 'cert']:
            self.adv_validate_epoch(epoch)
        # elif self.args.val_mode == 'prune':
        #     validate_epoch = self.prune_validate_epoch
        else:
            raise NameError
        return

    def train_step(self, images, labels):
        if self.args.train_mode == 'cert':
            self.normal_train_step(images, labels)
        elif self.args.train_mode == 'normal':
            self.normal_train_step(images, labels)
        elif self.args.train_mode == 'adv':
            self.adv_train_step(images, labels)
        # elif self.args.train_mode == 'prune':
        #     train_epoch = self.prune_train_epoch
        else:
            raise NameError
