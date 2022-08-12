import torch
from compute_jac import *
from settings.train_settings import *
from settings.test_setting import *
from engine.trainer import Trainer
from models.base_model import build_model
from matplotlib.pyplot import plot
if __name__ == '__main__':
    arg_var = ['--dir', 'distance/dnn_bn', '--net', 'dnn', '--dataset', 'mnist',
               '--train_mode', 'std', '--val_mode', 'std',
               '--epoch_type', 'step', '--total_step', '3', '--epoch_step', '3', '--warmup', 0,
               '--lr_schedule', 'cyclic']
    for depth in ['2', '3', '5', '9', '17']:
        for width in ['100', '200', '400', '800', '1600']:
            arg_var_cur = arg_var + ['--depth', depth, '--width', width, '--exp_id', '_'.join([depth, width])]
            arg_parser = ArgParser(True, arg_var_cur)
            trainer = Trainer(arg_parser.get_args())
            trainer.train_model()
            arg_var_cur += ['--batch_size', '1']
            args = set_up_testing(arg_var_cur)
            _, test_loader = set_loader(args)
            model = build_model(args)
            model.load_model(args.model_dir)
            model.eval()

            all_li_dis, all_l2_dis = [], []
            for idx, (img, label) in enumerate(test_loader):
                li_dis, l2_dis = [], []
                jac, diff = compute_jac_cnn(model.layers, img.cuda(), activation=args.activation)
                pre_act_hook = ModelHook(model, set_input_hook)
                pre = model(img.cuda())
                pre_activation = pre_act_hook.retrieve_res(unpack)
                pre_act_hook.remove()
                for block_pre_activation, block_jac in zip(pre_activation, jac[:-1]):
                    layer_distance = block_pre_activation[0][0]/to_numpy(block_jac.norm(p=float('inf'), dim=0))
                    li = np.abs(block_pre_activation[0] / to_numpy(block_jac.norm(p=float('inf'), dim=0)))
                    l2 = np.abs(block_pre_activation[0] / to_numpy(block_jac.norm(p=2, dim=0)))
                    li.sort()
                    l2.sort()
                    li_dis += [li[0, :20]]
                    l2_dis += [l2[0, :20]]
                all_li_dis += [li_dis]
                all_l2_dis += [l2_dis]

            torch.save({'li_dis': all_li_dis, 'l2_dis': all_l2_dis},
                       os.path.join(MODEL_PATH, args.dir, 'distance', args.exp_id + '.pth'))

            print(1)


