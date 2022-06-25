import datetime
from time import time

from core.smooth_core import *
from dataloader.base import *
from core.smooth_analyze import *


def smooth_pred(model, args):
    smoothed_classifier = Smooth(model, args.num_cls, 0.25)

    # prepare output file
    outfile = args.model_dir + 'smooth'
    f = open(outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    _, dataset = set_data_set(args)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % 1 != 0:
            continue
        if i == -1:
            break

        (x, label) = dataset[i]

        before_time = time.time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, 100, 10000, 0.001, 1000)
        after_time = time.time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()