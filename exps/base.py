from exps.smoothed_certify import *


def set_exp(args):
    if args.test_name == 'smooth':
        return smooth_test
    else:
        raise NameError
