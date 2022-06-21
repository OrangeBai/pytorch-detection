import logging
import datetime
from torch.nn.functional import one_hot
from models import *


class Trainer:
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
        if self.args.warmup_steps == 0:
            return
        loader = InfiniteLoader(self.train_loader)
        self.lr_scheduler = warmup_scheduler(self.args, self.optimizer)
        for cur_step in range(self.args.warmup_steps):
            images, labels = next(loader)
            images, labels = to_device(self.args.devices[0], images, labels)
            Trainer.train_step(self, images, labels)
            if cur_step % self.args.print_every == 0:
                self.step_logging(cur_step, self.args.warmup_steps, -1, self.args.num_epoch, loader.metric)

            if cur_step >= self.args.warmup_steps:
                break
        self.train_logging(-1, self.args.num_epoch, loader.metric)
        self.validate_epoch(-1)

        self.optimizer = init_optimizer(self.args, self.model)
        self.lr_scheduler = init_scheduler(self.args, self.model.optimizer)
        return

    def validate_epoch(self, epoch):
        start = time.time()
        self.model.eval()
        for images, labels in self.test_loader:
            images, labels = to_device(self.args.devices[0], images, labels)
            pred = self.model(images)
            top1, top5 = accuracy(pred, labels)
            self.update_metric(top1=(top1, len(images)), top5=(top5, len(images)))
        self.model.train()
        msg = self.val_logging(epoch) + '\ttime:{0:.4f}'.format(time.time() - start)

        self.logger.info(msg)
        print(msg)
        return

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)
        loss.backward()
        self.step()
        top1, top5 = accuracy(outputs, labels)
        self.update_metric(top1=(top1, len(images)), top5=(top5, len(images)),
                           loss=(loss, len(images)), lr=(self.get_lr(), 1))

    def step(self):
        self.optimizer.step()
        self.lr_scheduler.step()

    def update_metric(self, **kwargs):
        self.metrics.update(**kwargs)
        self.metrics.synchronize_between_processes()

    def train_epoch(self, epoch, *args, **kwargs):
        for step in range(self.args.epoch_step):
            images, labels = next(self.inf_loader)
            self.train_step(images, labels)
            if step % self.args.print_every == 0:
                self.step_logging(step, self.args.epoch_step, epoch, self.args.num_epoch, self.inf_loader.metric)
        self.train_logging(epoch, self.args.num_epoch, time_metrics=self.inf_loader.metric)
        self.inf_loader.reset()
        return

    def train_model(self):
        self.warmup()

        for epoch in range(self.args.num_epoch):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            self.record_result(epoch)

        self.model.save_model(self.args.model_dir)
        self.model.save_result(self.args.model_dir)
        return


def set_trainer(args):
    if args.train_mode == 'normal':
        trainer = Trainer(args)
    elif args.train_mode == 'cert':
        train_file_name = 'core.engine.cert_train'
        modules = importlib.import_module(train_file_name)
        trainer = modules.__dict__['CertTrainer'](args)

    return trainer



