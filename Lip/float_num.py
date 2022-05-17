import numpy as np
import torch.autograd.functional
from scipy.linalg import svd
from settings.test_setting import set_up_testing
from models.base_model import *
from core.pattern import *
from dataloader.base import *
from core.utils import *
from attack import *

if __name__ == '__main__':
    args = set_up_testing()
    model = BaseModel(args)

    args.num_of_est = [64, 128, 256, 512, 1024, 1536, 2048]
    args.noise = [4 / 255, 8 / 255, 12/255, 16 / 255, 24/255, 32 / 255, 64/255]
    model.load_model(args.model_dir)
    args.batch_size = 1
    train_loader, test_loader = set_loader(args)

    res = {}
    for num_of_est in args.num_of_est:
        for noise in args.noise:
            rate = []
            total = 1
            noise_attack = Noise(model.model, args.devices[0], noise, mean=(0.1307,), std=(0.3081,))
            batch_flt_hook = ModelHook(model.model, hook=retrieve_float_neuron_hook,
                                       Gamma=[0], batch_size=num_of_est)
            for idx, (img, label) in enumerate(test_loader):
                model.model.eval()
                noise_img = noise_attack.attack(img, batch_size=num_of_est, device=args.devices[0])
                noise_img.require_grad = True

                a = model.model(noise_img)
                cur_batch_min = batch_flt_hook.retrieve_res(unpack)

                if idx == 0:
                    total = np.array(cur_batch_min).size
                rate.append(np.array(cur_batch_min).sum() / total)
                if idx > 500:
                    break
                # avg = [cur_batch_min]
                # total.append(cu)
                print(idx)
            batch_flt_hook.reset()
            name = 'est_{0}_noise_{1}'.format(num_of_est, noise * 255)
            res[name] = rate
    np.save(os.path.join(args.model_dir, 'rate'), np.array(res))
print(1)
