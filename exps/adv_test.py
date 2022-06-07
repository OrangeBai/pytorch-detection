from settings.test_setting import *
from models.base_model import *
from exps.utils import *
from attack import *


def avd_test(args):
    model = BaseModel(args)
    model.load_model(args.model_dir)
    model.model.eval()
    _, test_loader = set_loader(args)
    mean, std = get_mean_std(args)
    atk_fgsm = FGSM(model.model, args.device[0], mean=mean, std=std)

    for images, label in test_loader:
        images = atk_fgsm.attack(images, label)
        pre = model.model(images)

        top1, top5 = accuracy(pre, label)

    return


if __name__ == '__main__':
    argvs = [
        ['--exp_id', 'l_0.05_b_0.1_e_0.8', '--batch_size', '1', '--net', 'vgg16', '--dataset', 'cifar100'],
        ['--exp_id', 'l_0.00_b_0.1_e_0.8', '--batch_size', '1', '--net', 'vgg16', '--dataset', 'cifar100'],
        ['--exp_id', 'l_-0.05_b_0.1_e_0.8', '--batch_size', '1', '--net', 'vgg16', '--dataset', 'cifar100']
    ]
    res = []

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
