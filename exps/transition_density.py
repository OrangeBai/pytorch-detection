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
    for x, y, z in zip(*loaders):
        line_data = straight_line_td(x[0], y[0], 1000)
        cks = [torch.concat(line_data[i:i + 256], dim=0) for i in range(0, len(line_data), 256)]
        for ck in cks:
            model.model(ck.cuda())
        pattern_id = pattern_hook.retrieve_res(unpack)

        a = [[np.concatenate(val, axis=0) for val in block.values()] for block in pattern_id]
        print(1)
    #     break
    #
    # for data, label in train_loader:
    #     print(1)
    #     break

    print(1)
