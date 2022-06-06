from settings.train_settings import *
import argparse


def set_up_testing(test_name='normal', argv=None):
    arg_parser = ArgParser(False, argv)
    parser = arg_parser.parser
    args = arg_parser.args
    if test_name == 'normal':
        pass
    elif test_name.lower() == 'aplip':
        parser = ap_lip(parser)
    elif test_name.lower() == 'td':
        parser = td(parser)
    else:
        raise NameError('test name {0} not found'.format(test_name))

    return parser.parse_args(args)


def ap_lip(parser):
    parser.add_argument('--epsilon', nargs='+', default=[2 / 255, 4 / 255, 8 / 255, 16 / 255], type=float)
    parser.add_argument('--sample_size', default=256, type=int)
    parser.add_argument('--num_test', default=100)
    return parser


def td(parser):
    parser.add_argument('--line_breaks', default=2048, type=int)
    parser.add_argument('--num_test', default=100, type=int)
    parser.add_argument('--pre_batch', default=512)
    return parser
