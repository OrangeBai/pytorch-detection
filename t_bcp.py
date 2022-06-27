from exps.non_returnable import *
from settings.test_setting import *
from plt.non_returnable import *
from smooth.core import *
from core.BCP import *
from core.BCP_utils import argparser
from core.smooth_analyze import *
from core.smooth import *
from models.net.cxfy import *
if __name__ == '__main__':
    argv = ['--exp_id', 'test_lip', '--net', 'cxfy42', '--dataset', 'cifar10', '--batch_norm', '0', '--activation', 'ReLU']
    args = set_up_testing('td', argv)
    state_dict = torch.load(r'/home/orange/Main/Experiment/cifar10/cifar_10_bcp/temporary_best.pth')['state_dict']
    model = build_model(args)
    model.load_weights(state_dict[0])
    model = model.eval()
    model = model.cuda()
    _, test_loader = set_loader(args)
    # args2 = argparser()
    # evaluate_BCP(test_loader, model, 32/255, -1, -1, -1, args2, None)
    metrics = MetricLogger()
    for images, labels in test_loader:
        images, labels = to_device(0, images, labels)
        pred = model(images)
        top1, top5 = accuracy(pred, labels)
        metrics.update(top1=(top1, len(images)), top5=(top5, len(images)))
    smooth_pred(model, args)
    ApproximateAccuracy(r'/home/orange/Main/Experiment/cifar10/cifar10_cxfy42_test_bcp_smooth')