from settings.train_settings import *
import argparse


def set_up_testing(test_name='normal'):
    parser = ArgParser(False).get_args()

    if test_name == 'normal':
        pass
    elif test_name.lower() == 'aplip':
        parser = ap_lip(parser)
    elif test_name.lower() == 'td':
        parser = td(parser)
    else:
        raise NameError('test name {0} not found'.format(test_name))

    return parser.parse_args()


def ap_lip(parser):
    parser.add_argument('--epsilon', nargs='+', default=[2 / 255, 4 / 255, 8 / 255, 16 / 255], type=float)
    parser.add_argument('--sample_size', default=256, type=int)
    parser.add_argument('--num_test', default=100)
    return parser


def td(parser):
    return parser
