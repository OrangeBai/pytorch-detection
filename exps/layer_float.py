import torch

from settings.train_settings import *
from engine import *
from settings.train_settings import *
from settings.test_setting import *
from models.base_model import build_model
from dataloader.base import set_data_set
from core.pattern import *

if __name__ == '__main__':
    argv = [
        '--net', 'vgg13', '--dataset', 'cifar10', '--dir', 'est_float', '--exp_id', 'std',
        '--data_bn', '1', '--activation', 'ReLU', '--batch_size', '256',
        '--epoch_type', 'epoch', '--num_epoch', '60', '--lr_scheduler', 'cyclic', '--warmup', '0',
        '--train_mode', 'std', '--val_mode', 'std'
    ]
    # arg_parser = ArgParser(True, argv)
    # for file_path in arg_parser.files:
    #     arg_parser.modify_parser(file_path)
    #
    #     trainer = Trainer(arg_parser.get_args())
    #     trainer.train_model()
    args = set_up_testing(argv)
    model = build_model(args)
    model.load_model(args.model_dir)
    model = model.eval()
    batch_size = 128
    eps = 32 / 255
    _, test_data_set = set_data_set(args)

    pattern_hook = ModelHook(model, set_pattern_hook, set_gamma(args.activation))
    pre_act_hook = ModelHook(model, set_input_hook)
    dataset_p = []
    shape = [32 * 32 * 3]
    for i in range(len(test_data_set)):
        x, y = test_data_set.__getitem__(i)
        x_batch = x.unsqueeze(dim=0).repeat((batch_size, 1, 1, 1)).cuda()
        noise = torch.randn_like(x_batch, device='cuda')
        noise_norm = noise.view(batch_size, -1).norm(p=2, dim=-1)
        normed_noise = noise / noise_norm.view((batch_size,) + (1,) * (len(noise.shape) - 1)) * eps
        x_batch = x_batch + normed_noise
        pred = model(x_batch)
        pattern = pattern_hook.retrieve_res(unpack)
        pre_act = pre_act_hook.retrieve_res(unpack)

        sample_p = []
        sample_shape = []
        for layer_pt in pattern:
            block_p = []
            block_shape = []
            for block_pt in layer_pt:
                diff = (block_pt - block_pt[0]) != 0
                p = diff[1:].mean() * 100
                block_p += [p]
                block_shape += [diff[0].size]
            sample_p += [block_p]
            sample_shape += [block_shape]
        if i == 0:
            shape += [sample_shape]
        dataset_p += [sample_p]
        if i % 50 == 0:
            print(1)
    print(1)
