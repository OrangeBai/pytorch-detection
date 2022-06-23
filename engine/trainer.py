from engine.adv_train import *
from engine.cert_train import CertTrainer


class Trainer(CertTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.train_epoch, self.validate_epoch = self.set_train_val()

    def train_model(self):
        self.warmup()
        self.validate_epoch(-1)
        for epoch in range(self.args.num_epoch):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            self.record_result(epoch)

        self.model.save_model(self.args.model_dir)
        self.model.save_result(self.args.model_dir)

    def set_train_val(self):
        if self.args.train_mode == 'cert':
            train_epoch = self.cert_train_epoch
        elif self.args.train_mode == 'normal':
            train_epoch = self.normal_train_epoch
        elif self.args.train_mode == 'adv':
            train_epoch = self.adv_train_epoch
        # elif self.args.train_mode == 'prune':
        #     train_epoch = self.prune_train_epoch
        else:
            raise NameError

        if self.args.val_mode in ['normal', 'adv', 'cert']:
            validate_epoch = self.adv_validate_epoch
        # elif self.args.val_mode == 'prune':
        #     validate_epoch = self.prune_validate_epoch
        else:
            raise NameError
        return train_epoch, validate_epoch
