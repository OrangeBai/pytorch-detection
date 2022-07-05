from core.utils import *
from dataloader.base import *
from models.base_model import build_model
from settings.test_setting import *


def _single_test(activation, net, batch_size):
    base_dir = os.path.join('100run', '_'.join([activation, str(batch_size)]))

    gen_result = []
    for i in range(5):
        exp_id = '_'.join(['gen', '0'+str(i)])
        argv = ['--dir', base_dir, '--exp_id', exp_id, '--net', net, '--dataset', 'cifar10', '--activation', activation]
        args = set_up_testing(argv)
        model = build_model(args)
        model.load_model(args.model_dir, 'cur_best')
        model.eval()
        _, test_loader = set_loader(args)
        metrics = MetricLogger()
        for images, labels in test_loader:
            outputs = model(images.cuda())
            top1, top5 = accuracy(outputs, labels.cuda())
            metrics.update(top1=(top1, args.batch_size))
        gen_result += [metrics.top1.global_avg]
    print(1)

    std_result = []
    for i in range(5):
        exp_id = '_'.join(['std', '0'+str(i)])
        argv = ['--dir', base_dir, '--exp_id', exp_id, '--net', net, '--dataset', 'cifar10', '--activation', activation]
        args = set_up_testing(argv)
        model = build_model(args)
        model.load_model(args.model_dir, 'cur_best')
        model.eval()
        _, test_loader = set_loader(args)
        metrics = MetricLogger()
        for images, labels in test_loader:
            outputs = model(images.cuda())
            top1, top5 = accuracy(outputs, labels.cuda())
            metrics.update(top1=(top1, args.batch_size))
        std_result += [metrics.top1.global_avg]
    return np.array(gen_result), np.array(std_result)


if __name__ == '__main__':
    gen_acc, std_acc = _single_test('LeakyReLU', 'cxfy42', 256)
    print('Gen Acc {0}, Var {1}'.format(gen_acc.mean(), gen_acc.var()))
    print('Std Acc {0}, Var {1}'.format(std_acc.mean(), std_acc.var()))
    print(1)
