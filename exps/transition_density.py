import numpy as np

from settings.test_setting import *
from models.base_model import *
from exps.utils import *


def cal_td(args):
    model = BaseModel(args)
    model.load_model(args.model_dir)
    model.model.eval()
    pattern_hook = ModelHook(model.model, retrieve_pattern, Gamma=[0])
    loaders = set_single_loaders(args, *[1, 2, 3])
    for data_idx, (x, y, z) in enumerate(zip(*loaders)):
        if data_idx == args.num_test:
            break
        line_data = straight_line_td(x[0], y[0], args.line_breaks)
        cks = [torch.concat(line_data[i:i + args.pre_batch], dim=0) for i in range(0, len(line_data), args.pre_batch)]
        for batch_idx, ck in enumerate(cks):
            pre = model.model(ck.cuda())

        pattern_id = pattern_hook.retrieve_res(unpack)
        pattern_id = [np.concatenate(val, axis=0) for block in pattern_id for val in block.values()]

        td_lst = []
        for layer_idx, layer in enumerate(pattern_id):
            df = np.abs(np.diff(layer, axis=0))
            td_lst.append(np.reshape(df, (df.shape[0], -1)).sum(axis=1))
        return np.array(td_lst).mean(axis=0)


if __name__ == '__main__':
    argvs = [
        ['--exp_id', 'l_0.05_b_0.1_e_0.8', '--batch_size', '1', '--net', 'vgg16', '--dataset', 'cifar100'],
        ['--exp_id', 'l_0.00_b_0.1_e_0.8', '--batch_size', '1', '--net', 'vgg16', '--dataset', 'cifar100'],
        ['--exp_id', 'l_-0.05_b_0.1_e_0.8', '--batch_size', '1', '--net', 'vgg16', '--dataset', 'cifar100']
    ]
    res = []
    for argv in argvs:
        args = set_up_testing(test_name='td', argv=argv)
        res.append(cal_td(args))

    # torch.save(td, os.path.join(args.model_dir), 'td')
    import matplotlib.pyplot as plt

    for i in range(len(argvs)):
        res[i] = [res[i][j: j + 16].sum() for j in range(0, len(res[i]), 16)]
    plt.plot(np.linspace(0, 1, len(res[0])), res[0])
    plt.plot(np.linspace(0, 1, len(res[0])), res[1])
    plt.plot(np.linspace(0, 1, len(res[0])), res[2])

    plt.plot()
    plt.legend(['005','000', '-005'])
    print(1)
    #     break
    #
    # for data, label in train_loader:
    #     print(1)
    #     break

    print(1)
