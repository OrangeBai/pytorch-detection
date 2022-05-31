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
    pattern_hook = ModelHook(model.model, retrieve_input_hook)
    train_loader, test_loader = set_loader(args)
    td = np.zeros((10, 5120))
    for data, label in test_loader:
        model.model(data.cuda())
        pattern_id = pattern_hook.retrieve_res(unpack)
        pattern_id = [np.concatenate(val, axis=0) for block in pattern_id for val in block.values()]
        for layer_idx, layer in enumerate(pattern_id):
            l_m = pattern_id[0][np.abs(pattern_id[0]) > 0].min()
            td[data_idx][l_m] = df
        counter += 1

    torch.save(td, os.path.join(args.model_dir),  'td')

    print(1)
    #     break
    #
    # for data, label in train_loader:
    #     print(1)
    #     break

    print(1)
