import torch.nn as nn
from core.utils import *
import importlib


class BaseModel(nn.Module):
    # TODO Record epoch info
    def __init__(self, args, logger):
        super(BaseModel, self).__init__()
        self.args = args
        self.model = build_model(args)
        self.optimizer = init_optimizer(args, self.model)
        self.lr_scheduler = init_scheduler(args, self.optimizer)
        self.loss_function = self.set_loss()

        self.logger = logger
        self.result = {'train': dict(), 'test': dict()}
        self.metrics = MetricLogger()

    @staticmethod
    def set_loss():
        # TODO add more losses
        return nn.CrossEntropyLoss()

    def save_model(self, path, name=None):
        if not name:
            model_path = os.path.join(path, 'weights.pth')
        else:
            model_path = os.path.join(path, 'weights_{}.pth'.format(name))
        torch.save(self.model.state_dict(), model_path)
        return

    def load_model(self, path, name=None):
        if not name:
            model_path = os.path.join(path, 'weights.pth')
        else:
            model_path = os.path.join(path, 'weights_{}.pth'.format(name))
        self.model.load_state_dict(torch.load(model_path), strict=False)
        return

    def save_result(self, path, name=None):
        if not name:
            res_path = os.path.join(path, 'result')
        else:
            res_path = os.path.join(path, 'result_{}'.format(name))
        np.save(res_path, self.result)

    def train_step(self, images, labels):
        images, labels = to_device(self.args.devices[0], images, labels)
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        top1, top5 = accuracy(outputs, labels)
        self.metrics.update(top1=(top1, len(images)), loss=(loss, len(images)),
                            lr=(self.optimizer.param_groups[0]['lr'], 1))
        self.metrics.synchronize_between_processes()
        return

    def record_result(self, epoch, mode='train'):

        epoch_result = {}
        for k, v in self.metrics.meters.items():
            epoch_result[k] = v.to_dict()
        self.result[mode][epoch] = epoch_result
        self.metrics.reset()
        return

    def validate_model(self, epoch, test_loader):
        # TODO validation logging
        self.model.eval()
        for images, labels in test_loader:
            images, labels = to_device(self.args.devices[0], images, labels)
            pred = self.model(images)
            top1, top5 = accuracy(pred, labels)
            self.metrics.update(top1=(top1, len(images)))
        return

    def logging(self, step, batch_num, epoch, epoch_num, time_metrics):
        # TODO maybe a refactor???
        space_fmt = ':' + str(len(str(batch_num))) + 'd'

        log_msg = '  '.join(['Epoch: [{epoch}/{epoch_num}]',
                             '[{step' + space_fmt + '}/{batch_num}]',
                             '{time_str}',
                             '{meters}',
                             '{memory}'
                             ])

        eta_seconds = time_metrics.meters['iter_time'].global_avg * (batch_num - step)
        eta_string = 'eta: {}'.format(str(datetime.timedelta(seconds=int(eta_seconds))))

        time_str = '\t'.join([eta_string, str(time_metrics)])

        msg = log_msg.format(epoch=epoch, epoch_num=epoch_num,
                             step=step, batch_num=batch_num,
                             time_str=time_str, meters=str(self.metrics),
                             memory='max mem: {0:.2f}'.format(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
                             )

        return msg

    def epoch_logging(self, epoch, epoch_num, time_metrics=None):
        """
        Print loggings after training of each epoch
        @param epoch:
        @param epoch_num:
        @param time_metrics:
        @return:
        """
        log_msg = '  '.join(['Epoch: [{epoch}/{epoch_num}] training finished\n',
                             'Training information:', '{meters}', ])

        msg = log_msg.format(epoch=epoch, epoch_num=epoch_num, meters=str(self.metrics))
        if time_metrics is not None:
            msg += '\ttime: {time:.4f}'.format(time=time_metrics.meters['iter_time'].total)

        self.record_result(epoch)
        return msg

    def val_logging(self, epoch):
        log_msg = '\n'.join(['Validationn Informtion:', '{meters}']).format(self.metrics)
        self.record_result(epoch, 'test')
        return log_msg


def build_model(args):
    """Import the module "model/[model_name]_model.py"."""
    model_file_name = "models." + args.model_type
    modules = importlib.import_module(model_file_name)
    model = None
    for name, cls in modules.__dict__.items():
        if name.lower() in args.net.lower():
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            model_file_name, args.net))
        exit(0)
    else:
        return to_device(args.devices[0], model(args))[0]
