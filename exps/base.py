from exps.smoothed_certify import smooth_test

def set_exp(args):
    if args.test_name == 'smooth':
        return smooth_test
    elif args.test_name == '100run':
        return None
