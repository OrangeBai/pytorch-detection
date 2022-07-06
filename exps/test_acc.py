from dataloader.base import *
from attack import set_attack
from core.utils import to_device, accuracy, MetricLogger

def test_acc(model, args):
    _, test_loader = set_loader(args)
    mean, std = set_mean_sed(args)
    attack_args = {
        'mean': mean,
        'std': std,
        # 'mean': [0,0,0],
        # 'std': [1,1,1],
        'eps': args.eps,
        'alpha': args.alpha,
        'ord': args.ord
    }
    atks = {'fgsm':set_attack(model, 'FGSM', args.devices[0], **attack_args),
            'pgd': set_attack(model, 'PGD', args.devices[0], **attack_args)}
    metrics = MetricLogger()
    for images, labels in test_loader:
        images, labels = to_device(args.devices[0], images, labels)
        pred = model(images)
        top1, top5 = accuracy(pred, labels)
        metrics.update(top1=(top1, len(images)), top5=(top5, len(images)))
        for name, attack in atks.items():
            adv = attack.attack(images, labels)
            pred_adv = model(adv)
            top1, top5 = accuracy(pred_adv, labels)
            update_times = {name + 'top1': (top1, len(images))}
            metrics.update(**update_times)
    return metrics
