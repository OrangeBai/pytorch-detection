from settings.test_setting import *
from models.base_model import *
from exps.utils import *
from attack import *


def avd_test(args):
    model = BaseModel(args)
    model.load_model(args.model_dir)
    model.model.eval()
    _, test_loader = set_loader(args)
    mean, std = set_mean_sed(args)
    atk_fgsm = FGSM(model.model, args.devices[0], eps=2/255, mean=mean, std=std)

    metric = MetricLogger()
    for images, labels in test_loader:
        images, labels = to_device(args.devices[0], images, labels)
        adv_images = atk_fgsm.attack(images, labels)
        pre_ori = model.model(images)
        pre_adv = model.model(adv_images)

        top1, top5 = accuracy(pre_ori, labels)
        metric.update(top1=(top1, args.batch_size), top5=(top5, args.batch_size))

        top1, top5 = accuracy(pre_adv, labels)
        metric.update(top1_avd=(top1, args.batch_size), top5_adv=(top5, args.batch_size))

    return metric


if __name__ == '__main__':
    argvs = [
        ['--exp_id', 'l_0.00_b_0.1_e_0.8', '--batch_size', '128', '--net', 'vgg16', '--dataset', 'cifar100'],
        ['--exp_id', 'l_0.05_b_0.1_e_0.8', '--batch_size', '128', '--net', 'vgg16', '--dataset', 'cifar100'],
        # ['--exp_id', 'l_-0.05_b_0.1_e_0.8', '--batch_size', '128', '--net', 'vgg16', '--dataset', 'cifar100']
    ]
    res = []
    for argv in argvs:
        args = set_up_testing('normal', argv)
        res.append(avd_test(args))
    # torch.save(td, os.path.join(args.model_dir), 'td')
    import matplotlib.pyplot as plt

    # for i in range(len(argvs)):
    #     res[i] = [res[i][j: j + 16].sum() for j in range(0, len(res[i]), 16)]
    # plt.plot(np.linspace(0, 1, len(res[0])), res[0])
    # plt.plot(np.linspace(0, 1, len(res[0])), res[1])
    # plt.plot(np.linspace(0, 1, len(res[0])), res[2])

    plt.plot()
    plt.legend(['005', '000', '-005'])
    print(1)
    #     break
    #
    # for data, label in train_loader:
    #     print(1)
    #     break

    print(1)
