from plt.plt_funs import *
from settings.test_setting import *
import numpy as np
if __name__ == '__main__':
    fig, ax = plt.subplots()

    labels = ['100', '200', '400', '800']
    ax.set_xticklabels(labels)
    for depth in ['2', '3', '5', '9', '17']:
        li_min = []
        l2_min = []
        for width in ['100', '200', '400', '800', '1600']:
            arg_var = ['--dir', 'distance/dnn_bn', '--net', 'dnn', '--dataset', 'mnist',
                       '--exp_id', '_'.join([depth, width])]
            args = set_up_testing(arg_var)
            data = torch.load(os.path.join(MODEL_PATH, 'distance/dnn_bn', 'distance', args.exp_id + '.pth'))

            li_min += [np.log(np.array(data['li_dis']).mean(axis=0).min(axis=0)[0])]
            l2_min += [np.log(np.array(data['l2_dis']).mean(axis=0).min(axis=0)[0])]
        ax.plot(['100', '200', '400', '800'], li_min)


    print(1)
