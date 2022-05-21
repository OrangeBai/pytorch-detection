import yaml

from settings.base_funs import *
import argparse


def set_up_testing():
    parser = argparse.ArgumentParser()
    args = sys.argv[1:]
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--net', default='dnn')
    parser.add_argument('--exp_id', default='0')
    parser.add_argument('-cuda', default=[0])

    parser.add_argument('--epsilon', nargs='+', default=[2 / 255, 4 / 255, 8 / 255, 16 / 255], type=float)
    parser.add_argument('--sample_size', default=256, type=int)
    parser.add_argument('--num_test', default=100)

    parser = model_dir(parser, False)
    parser = devices(parser)

    cur_args, _ = parser.parse_known_args()
    # Load configuration from yaml file
    with open(os.path.join(cur_args.model_dir, 'args.yaml'), 'r') as file:
        args_dict = yaml.load(file, Loader=yaml.FullLoader)

    for key, val in args_dict.items():
        if key not in vars(cur_args).keys():
            parser.add_argument('--' + key, default=val, type=type(val))

    return parser.parse_args(args)


if __name__ == '__main__':
    set_up_testing()
