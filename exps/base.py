from exps.smoothed_certify import smooth_test
from exps.test_acc import test_acc

def set_exp(args):
    if args.test_name == 'smooth':
        return smooth_test
    elif args.test_name == '100run':
        return None
    elif args.test_name == 'test_acc':
        return test_acc
