from settings.test_setting import set_up_testing
from models.base_model import *
from core.pattern import *
from dataloader.base import *
from attack import *
from Lip.utils import *

if __name__ == '__main__':
    argv = ['--exp_id', 'l_0.00_b_0.1_e_0.8', '--batch_size', '1', '--net', 'vgg16', '--dataset', 'cifar100']
    args = set_up_testing('aplip', argv)
    model = BaseModel(args)
    model.load_model(args.model_dir)
    train_loader, test_loader = set_loader(args)
    # TODO refactor noise module
    noise_attack = Noise(model.model, args.devices[0], args.epsilon[0], mean=(0.5,), std=(1,))

    float_hook = ModelHook(model.model, hook=set_pattern_hook, Gamma=[0])
    # Record all the weight matrix

    block_weights, block_types = record_blocks(model)
    r = 1
    avg = []
    t = time.time()
    for idx, (img, label) in enumerate(test_loader):
        model.model.eval()
        noise_img = noise_attack.attack(img, batch_size=args.sample_size, device=args.devices[0])
        noise_img.require_grad = True

        a = model.model(noise_img)
        cur_lb_ub = float_hook.retrieve_res(retrieve_lb_ub, reset=False, grad_bound=[(0,0), (1,1)], sample_size=64)
        cur_float = float_hook.retrieve_res(retrieve_float_neurons, reset=True, sample_size=64, )
        if args.net == 'dnn':
            loc_jac = compute_jac(cur_lb_ub, block_weights, block_types, batch_size=args.batch_size)
            loc_lip = [svd(jac)[1][0] for jac in loc_jac]
        else:
            loc_lip = np.ones(args.batch_size)

        # r = np.ones(args.batch_size)
        # linear_counter = 0
        # for block_weight, block_type, block_float in zip(block_weights[:-1], block_types[:-1], cur_float[:-1]):
        #     r *= amplify_ratio(block_float, block_weight, block_type)
        # ub_lb_hook.reset()
        # float_hook.reset()

        est = loc_lip * r
        avg += [est]
        # pre = model.model(img.cuda())[0]
        # margin = (pre.sort()[0][-1] - pre.sort()[0][-2]).cpu().detach().numpy()
        # atk = FGSM(model.model, args.devices[0],  eps=noise / est, mean=(0.5,), std=(1,))
        # # atk = PGD(model.model, mean, std, ee)
        # x = atk.attack(img.cuda(), label.cuda())
        # if torch.argmax(model.model(x)) == torch.argmax(pre):
        #     correct += [1]
        if len(avg) > 1:
            print(np.array(avg).mean())
            break

    print((time.time() - t) / 10)
    print(1)
#             EW1 = layer.weight[region_diff[layer_id]].cpu().detach().numpy()
#             L = layer.weight[x_pattern[layer_id] == 0].cpu().detach().numpy()
