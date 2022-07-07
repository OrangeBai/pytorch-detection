import datetime
import logging

from attack import set_attack
from dataloader.base import *
from models import *


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.model = build_model(args)

        self.mean, self.std = set_mean_sed(args)
        self.train_loader, self.test_loader = set_loader(args)
        self.inf_loader = InfiniteLoader(self.train_loader)

        self.optimizer = init_optimizer(args, self.model)
        self.lr_scheduler = init_scheduler(args, self.optimizer)

        self.loss_function = init_loss(args)

        self.metrics = MetricLogger()
        self.result = {'train': dict(), 'test': dict()}
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=os.path.join(args.model_dir, 'logger'))
        self.logger.info(args)

        self.lip = set_attack(self.model, 'Lip', args.devices[0], ord=args.ord)

        self.attack_args = {
            'mean': self.mean,
            'std': self.std,
            # 'mean': [0,0,0],
            # 'std': [1,1,1],
            'eps': self.args.eps,
            'alpha': self.args.alpha,
            'ord': self.args.ord
        }
        self.attacks = self.set_attack()

    def set_attack(self):
        return {'FGSM': set_attack(self.model, 'FGSM', self.args.devices[0], **self.attack_args),
                'PGD': set_attack(self.model, 'PGD', self.args.devices[0], **self.attack_args),
                'FFGSM': set_attack(self.model, 'FFGSM', self.args.devices[0], **self.attack_args)
                }

    def save_result(self, path, name=None):
        if not name:
            res_path = os.path.join(path, 'result')
        else:
            res_path = os.path.join(path, 'result_{}'.format(name))
        np.save(res_path, self.result)

    def record_result(self, epoch, mode='train'):

        epoch_result = {}
        for k, v in self.metrics.meters.items():
            epoch_result[k] = v.to_dict()
        self.result[mode][epoch] = epoch_result
        self.metrics.reset()
        return

    def step_logging(self, step, batch_num, epoch, epoch_num, time_metrics=None):
        space_fmt = ':' + str(len(str(batch_num))) + 'd'

        log_msg = '  '.join(['Epoch: [{epoch}/{epoch_num}]',
                             '[{step' + space_fmt + '}/{batch_num}]',
                             '{time_str}',
                             '{meters}',
                             '{memory}'
                             ])

        if time_metrics is not None:
            eta_seconds = time_metrics.meters['iter_time'].global_avg * (batch_num - step)
            eta_string = 'eta: {}'.format(str(datetime.timedelta(seconds=int(eta_seconds))))

            time_str = '\t'.join([eta_string, str(time_metrics)])
        else:
            time_str = ''

        msg = log_msg.format(epoch=epoch, epoch_num=epoch_num,
                             step=step, batch_num=batch_num,
                             time_str=time_str, meters=str(self.metrics),
                             memory='max mem: {0:.2f}'.format(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
                             )
        self.logger.info(msg)
        print(msg)
        return

    def train_logging(self, epoch, epoch_num, time_metrics=None):
        """
        Print loggings after training of each epoch
        @param epoch:
        @param epoch_num:
        @param time_metrics:
        @return:
        """
        self.logger.info('Epoch: [{epoch}/{epoch_num}] training finished'.format(epoch=epoch, epoch_num=epoch_num))
        log_msg = '\t'.join(['TRN INF:', '{meters}\t'])
        msg = log_msg.format(meters=str(self.metrics))
        if time_metrics is not None:
            msg += 'time: {time:.4f}'.format(time=time_metrics.meters['iter_time'].total)

        self.record_result(epoch)
        self.logger.info(msg)
        print(msg)
        return

    def val_logging(self, epoch):
        msg = '\t'.join(['VAL INF:', '{meters}']).format(meters=self.metrics)
        self.record_result(epoch, 'test')
        return msg

    @property
    def trained_ratio(self):
        return self.lr_scheduler.last_epoch / self.args.total_step

    def warmup(self):
        self.inf_loader.reset()
        if self.args.warmup_steps == 0:
            return
        loader = InfiniteLoader(self.train_loader)
        self.lr_scheduler = warmup_scheduler(self.args, self.optimizer)
        for cur_step in range(self.args.warmup_steps):
            images, labels = next(loader)
            images, labels = to_device(self.args.devices[0], images, labels)
            self.std_train_step(images, labels)
            if cur_step % self.args.print_every == 0 and cur_step != 0:
                self.step_logging(cur_step, self.args.warmup_steps, -1, self.args.num_epoch, loader.metric)

            if cur_step >= self.args.warmup_steps:
                break
        self.train_logging(-1, self.args.num_epoch, loader.metric)
        # self.validate_epoch(-1)
        self.optimizer = init_optimizer(self.args, self.model)
        self.lr_scheduler = init_scheduler(self.args, self.optimizer)

        return

    def std_validate_epoch(self, epoch):
        start = time.time()
        self.model.eval()
        for images, labels in self.test_loader:
            images, labels = to_device(self.args.devices[0], images, labels)
            pred = self.model(images)
            top1, top5 = accuracy(pred, labels)
            self.update_metric(top1=(top1, len(images)))
            if self.args.record_lip:
                self.record_lip(images, labels, pred)
        acc = self.metrics.top1.global_avg
        msg = self.val_logging(epoch) + '\ttime:{0:.4f}'.format(time.time() - start)
        self.logger.info(msg)
        print(msg)

        self.model.train()
        return acc

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def std_train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)

        self.step(loss)
        top1, top5 = accuracy(outputs, labels)
        self.update_metric(top1=(top1, len(images)), top5=(top5, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1))

    def record_lip(self, images, labels, outputs):
        perturbation = self.lip.attack(images, labels)
        local_lip = (self.model(images + perturbation) - outputs)
        lip_li = (local_lip.norm(p=float('inf'), dim=1) / perturbation.norm(p=float('inf'), dim=(1, 2, 3))).mean()
        lip_l2 = (local_lip.norm(p=2, dim=1) / perturbation.norm(p=2, dim=(1,2,3))).mean()
        self.update_metric(lip_li=(lip_li, len(images)), lip_l2=(lip_l2, len(images)))
        return

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    def update_metric(self, **kwargs):
        self.metrics.update(**kwargs)
        self.metrics.synchronize_between_processes()

    def std_train_epoch(self, epoch, *args, **kwargs):
        for step in range(self.args.epoch_step):
            images, labels = next(self.inf_loader)
            self.std_train_step(images, labels)
            if step % self.args.print_every == 0 and step != 0:
                self.step_logging(step, self.args.epoch_step, epoch, self.args.num_epoch, self.inf_loader.metric)
        self.train_logging(epoch, self.args.num_epoch, time_metrics=self.inf_loader.metric)

        return

    def train_epoch(self, epoch):
        pass

    def validate_epoch(self, epoch):
        pass

    def train_step(self, images, labels):
        pass
