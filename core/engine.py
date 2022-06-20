import logging
from torch.nn.functional import one_hot
from Lip.utils import *
from attack import *
from dataloader.base import *
from models import *


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = set_model(args)
        self.train_loader, self.test_loader = set_loader(args)
        self.result = {'train': dict(), 'test': dict()}
        self.metrics = MetricLogger()
        self.optimizer = init_optimizer(args, self.model)
        self.lr_scheduler = init_scheduler(args, self.optimizer)

        self.loss_function = init_loss(args)

        self.local_lip = None

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=os.path.join(args.model_dir, 'logger'))
        self.logger.info(args)

    def train_logging(self, step, batch_num, epoch, epoch_num, time_metrics=None):
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

    def epoch_logging(self, epoch, epoch_num, time_metrics=None):
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

    def trained_ratio(self):
        return self.lr_scheduler.last_epoch / self.args.total_step

    def warmup(self):
        loader = InfiniteLoader(self.train_loader)
        self.lr_scheduler = warmup_scheduler(self.args, self.optimizer)
        for cur_step in range(self.args.warmup_steps):
            images, labels = next(loader)
            images, labels = to_device(self.args.devices[0], images, labels)
            if self.args == 'cert':
                self.cert_train_step(images, labels)
            else:
                self.train_step(images, labels)
            if cur_step % self.args.print_every == 0:
                self.model.train_logging(cur_step, self.args.warmup_steps, -1, self.args.num_epoch, loader.metric)

            if cur_step >= self.args.warmup_steps:
                break
        self.model.optimizer = init_optimizer(self.args, self.model)
        self.model.lr_scheduler = init_scheduler(self.args, self.model.optimizer)
        return

    def cert_train_step(self, images, labels):
        pass

    def train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        top1, top5 = accuracy(outputs, labels)
        self.model.metrics.update(top1=(top1, len(images)), top5=(top5, len(images)), loss=(loss, len(images)),
                                  lr=(self.optimizer.param_groups[0]['lr'], 1))
        self.metrics.synchronize_between_processes()

    def train_epoch(self, args, model, inf_loader, cur_epoch):
        for cur_step in range(args.epoch_step):
            images, labels = next(inf_loader)
            self.train_step(images, labels)
        model.epoch_logging(cur_epoch, args.num_epoch, time_metrics=inf_loader.metric)
        inf_loader.reset()
        return

    def train_model(self):
        self.warmup()
        # logging.info(warmup(model, InfiniteLoader(train_loader)))
        # validate_model(model, -1, test_loader, True, alpha=1 / 255, eps=1.72 / 255, steps=7, restart=2)
        inf_loader = InfiniteLoader(self.train_loader)

        for cur_epoch in range(self.args.num_epoch):
            train_epoch(self.args, self.model, inf_loader, cur_epoch)
            # if cur_epoch % 10 == 0 and cur_epoch != 0:
            #     model.pruning_val(cur_epoch, test_loader)
            # else:
            validate_model(self.model, -1, self.test_loader, True, alpha=0.57 / 4 / 255, eps=0.57 / 255, steps=7)

        self.model.save_model(self.args.model_dir)
        self.model.save_result(self.args.model_dir)
        return


def train_epoch(args, model, inf_loader, cur_epoch):
    lip_mean = []
    for cur_step in range(args.epoch_step):
        images, labels = next(inf_loader)
        # model.train_step(images, labels)
        images, labels = to_device(model.args.devices[0], images, labels)
        model.optimizer.zero_grad()
        mean, std = set_mean_sed(model.args)

        lip = LipAttack(model.model, model.args.devices[0], eps=1 / 255, mean=mean, std=std)

        # if cur_step % 10 == 0 or cur_step == 0:
        # lip_ratio = torch.tensor(estimate_lip(model, images, 16)).view(len(images), 1).repeat(1, 10).cuda()
        # lip_mean += [lip_ratio.cpu()]
        # else:
        # lip_ratio = torch.stack(lip_mean).mean()
        perturbation = lip.attack(images, labels)
        # r = torch.randn(images.shape).cuda()
        # perturbation = images + r / r.norm(p=2, dim=(1, 2, 3)).view(len(r), 1, 1, 1) * 0.01
        outputs = model(images)
        certified_res = model(images + perturbation) - outputs
        c2 = certified_res.abs().max(axis=1)[0].view(len(certified_res), 1).repeat(1, 10)
        local_lip = (1 - one_hot(labels, num_classes=model.args.num_cls)).mul(c2).abs() * 1000 * 0.86
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
            model.train_logging(cur_step, args.epoch_step, cur_epoch, args.num_epoch, inf_loader.metric)

    model.epoch_logging(cur_epoch, args.num_epoch, time_metrics=inf_loader.metric)
    inf_loader.reset()


# def cert_train_step(model, images, labels):
#     return lip_ratio.mean()


def validate_model(model, epoch, test_loader, robust=False, *args, **kwargs):
    start = time.time()
    model.eval()
    mean, std = set_mean_sed(model.args)
    if robust:
        fgsm = set_attack(model, 'FGSM', model.args.devices[0], mean=mean, std=std, *args, **kwargs)
        pgd = set_attack(model, 'PGD', model.args.devices[0], mean=mean, std=std, *args, **kwargs)
    for images, labels in test_loader:
        images, labels = to_device(model.args.devices[0], images, labels)
        pred = model.model(images)
        top1, top5 = accuracy(pred, labels)
        model.metrics.update(top1=(top1, len(images)), top5=(top5, len(images)))
        if robust:
            adv = fgsm.attack(images, labels)
            pred_adv = model.model(adv)
            fgsm_top1, fgsm_top5 = accuracy(pred_adv, labels)
            adv = pgd.attack(images, labels)
            pred_adv = model.model(adv)
            pgd_top1, pgd_top5 = accuracy(pred_adv, labels)
            model.metrics.update(fgsm_top1=(fgsm_top1, len(images)), fgsm_top5=(fgsm_top5, len(images)),
                                 pgd_top1=(pgd_top1, len(images)), pgd_top5=(pgd_top5, len(images)))

    model.train()
    msg = model.val_logging(epoch) + '\ttime:{0:.4f}'.format(time.time() - start)

    model.logger.info(msg)
    print(msg)
    return
