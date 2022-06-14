from dataloader.base import *
from exps.utils import *
from models.base_model import *


def non_returnable_boundary(model, args):
    single_loaders = set_single_loaders(args, *[0, 1])

    pattern_hook = ModelHook(model, set_pattern_hook, [0])

    data_cg = []
    data_points = []
    for data_idx, ((x, _), (y, _)) in enumerate(zip(*single_loaders)):
        if data_idx == args.num_test:
            break
        line_data = straight_line_td(x[0], y[0], args.line_breaks)
        cks = [torch.stack(line_data[i:i + args.pre_batch], dim=0) for i in range(0, len(line_data), args.pre_batch)]
        print(1)

        for batch_idx, ck in enumerate(cks):
            pre = model.model(ck.cuda())
            if batch_idx > 10:
                break
        pattern_id = pattern_hook.retrieve_res(unpack)
        all_changed = []
        all_change_points = []
        for block in pattern_id:
            for layer in block:
                # where the pattern different from start point
                diff = np.reshape(layer - layer[0], (args.line_breaks, -1)) != 0
                recorder = np.zeros(diff.shape)
                status = np.zeros(diff.shape[1])
                change_point = np.zeros((2, diff.shape[1]))
                for i in range(args.line_breaks):
                    # where the pattern different from start point and never changed before
                    changed = np.where(np.all([diff[i], status == 0], axis=0))
                    status[changed] = 1
                    change_point[0, changed] = i
                    # where pattern changed and same with start point
                    returned = np.where(np.all([status == 1, ~diff[i]], axis=0))
                    status[returned] = 2
                    recorder[i] = status
                    change_point[1, returned] = i
                all_change_points.append(change_point)
                all_changed.append([np.sum(recorder == 1, axis=1), np.sum(recorder == 2, axis=1)])
        data_points.append(all_change_points)
        data_cg.append(all_changed)
    torch.save(np.array(data_cg), os.path.join(args.model_dir, 'exp', 'data_cg'))
    torch.save(data_points, os.path.join(args.model_dir, 'exp', 'data_points'))
    return data_cg
