import numpy as np
import torch
from settings.test_setting import *
from models.base_model import *
from exps.utils import *
if __name__ == '__main__':
    args = set_up_testing()
    model = BaseModel(args)
    model.load_model(args.model_dir)
    model.model.eval()
    pattern_hook = ModelHook(model.model, retrieve_pattern, Gamma=[0])
    loaders = set_single_loaders(args, *[1, 2, 3])
    train_loader, test_loader = set_loader(args)
    td = np.zeros((10, 5120))
    for data_idx, (x, y, z) in enumerate(zip(*loaders)):
        if data_idx == 10:
            break
        line_data = straight_line_td(x[0], y[0], 5120)
        cks = [torch.concat(line_data[i:i + 512], dim=0) for i in range(0, len(line_data), 512)]
        counter = 1
        for ck in cks:
            pre = model.model(ck.cuda())
            pattern_id = pattern_hook.retrieve_res(unpack)

            pattern_id = [np.concatenate(val, axis=0) for block in pattern_id for val in block.values()]
            for i in range(1, 512):
                df = 0
                for layer_idx, layer in enumerate(pattern_id):
                    df += ((layer[i] - layer[i-1]) != 0).sum()
                    td[data_idx][counter] = df
                counter += 1

    torch.save(td, os.path.join(args.model_dir),  'td')

    print(1)
    #     break
    #
    # for data, label in train_loader:
    #     print(1)
    #     break

    print(1)
