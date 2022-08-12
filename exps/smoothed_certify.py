import datetime
import os

import torch.nn as nn

from core.smooth_analyze import *
from core.smooth_core import *
from dataloader.base import *


def smooth_test(model, args):
    model.load_model(args.model_dir)
    model = model.eval()
    file_path = os.path.join(args.exp_dir, 'smooth')
    # mean, std = set_mean_sed(args)
    # model = nn.Sequential(*[Normalize(mean, std), model]).cuda()
    smooth_pred(model, args)

    certify_res = ApproximateAccuracy(file_path).at_radii(np.linspace(0, 1, 256))
    output_path = os.path.join(args.exp_dir, 'certify.npy')
    print(certify_res.mean())
    np.save(output_path, certify_res)
    return


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, images):
        # Broadcasting
        mean = self.mean.reshape(1, self.mean.shape[0], 1, 1)
        std = self.std.reshape(1, self.mean.shape[0], 1, 1)
        return (images - mean) / std


def smooth_pred(model, args):
    smoothed_classifier = Smooth(model, args.num_cls, args.sigma, args)

    # prepare output file
    outfile = os.path.join(args.exp_dir, 'smooth')
    f = open(outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    _, dataset = set_data_set(args)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % 50 != 0:
            continue
        if i == -1:
            break

        (x, label) = dataset[i]

        before_time = time.time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.smooth_alpha, args.batch)
        after_time = time.time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
