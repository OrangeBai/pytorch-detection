import argparse
import logging
import numpy as np
import random
import time

import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR

from attacks import *
from nets.preact_resnet import PreActResNet18
from cifar10.cifar import *
from core.utils import *
from copy import deepcopy

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])

    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])

    parser.add_argument('--net', default='resnet18', help='net name')
    parser.add_argument('--attack', default='FFGSM', help='name of the attack')
    parser.add_argument('--NAT', default='dynamic', choices=['dynamic', 'static', 'normal'])
    parser.add_argument('--noise', default='random', choices=['random', 'none'])
    parser.add_argument('--multiple-exp', default=0, type=int)
    parser.add_argument('--exp_id', default=0, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    data_dir, out_dir, logfile, model_path = set_up_exp(args)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    yml_file = load_yml('cifar10/config_cifar10.yml')
    logger.info(args)
    logger.info(yml_file)

    train_loader, test_loader = get_loaders(data_dir, args.batch_size, args.dataset)
    mean, std = get_mean_std(args.dataset)
    lower_limit, upper_limit = get_limits(mean, std)

    model = PreActResNet18().cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), **yml_file['Optimizer']['SGD'])
    lr_steps = args.epochs * len(train_loader) // 2
    if args.lr_schedule == 'cyclic':
        scheduler = CyclicLR(opt, step_size_up=lr_steps, step_size_down=lr_steps, **yml_file['Scheduler']['cyclic'])
    elif args.lr_schedule == 'multistep':
        yml_file['Scheduler']['cyclic']['miilstones'] = \
            [milestone * args.epochs for milestone in yml_file['Scheduler']['cyclic']['milestones']]
        scheduler = MultiStepLR(opt, **yml_file['scheduler']['multistep'])
    else:
        raise ModuleNotFoundError('No scheduler named {0}'.format(args.lr_schedule))
    nat_scheduler = get_nat_scheduler(args, yml_file)

    criterion = nn.CrossEntropyLoss()

    atk = get_attack(model, args.attack, mean, std)
    # Training
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    record = np.zeros((8, args.epochs))
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 1
        threshold = nat_scheduler(epoch)
        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()

            r = random.random()
            if r < threshold:
                x = atk.attack(x, y)
            else:
                if args.noise == 'random' and args.attack != 'VANILA':
                    delta = init_delta(x, atk.eps / atk.std)
                    delta.data = clamp(delta, lower_limit - x, upper_limit - x)
                    delta.requires_grad = True
                    x = x + delta
                else:
                    x = x
            output = model(x)
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            if r < threshold:
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)

        model_test = deepcopy(model)

        atk_cls = get_attack(model_test, 'FGSM', mean, std)
        record[0, epoch], record[1, epoch] = evaluate_attack(model_test, atk_cls, test_loader)

        atk_cls = get_attack(model_test, 'PGD', mean, std)
        record[6, epoch], record[7, epoch] = evaluate_attack(model_test, atk_cls, test_loader)

        record[2, epoch], record[3, epoch] = evaluate_attack(model_test, None, test_loader)

        record[4, epoch] = train_loss / train_n
        record[5, epoch] = train_acc / train_n

        # scheduler.step(epoch)
        epoch_time = time.time()
        lr = opt.param_groups[0]['lr']
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_acc / train_n)

    train_time = time.time()
    torch.save(model.state_dict(), model_path)
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time) / 60)
    np.save(os.path.join(out_dir, 'record.npy'), record)
    logger.info('Evalutating')

    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(torch.load(model_path))
    model_test.float()
    model_test.eval()
    atks = ['Basic', 'FGSM', 'BIM', 'RFGSM', 'PGD', 'FFGSM', 'TPGD', 'MIFGSM', 'APGD', 'PGDDLR']
    for atk in atks:
        t1 = time.time()
        if atk == 'Basic':
            loss, acc = evaluate_attack(model_test, None, test_loader)
        else:
            atk_cls = get_attack(model_test, atk, mean, std)
            loss, acc = evaluate_attack(model_test, atk_cls, test_loader)
        log_info = '{0} ACC: {1:.4f}\tLoss:{2:.4f}'.format(atk, acc, loss)
        logger.info(log_info)
        print(time.time() - t1)


if __name__ == "__main__":
    main()
