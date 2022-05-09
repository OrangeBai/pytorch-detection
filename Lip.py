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

    model.load_model(args.model_dir)
    single_flt_hk = ModelHook(model.model, hook=retrieve_float_hook,
                              Gamma=[0], grad_bound=((0, 0), (1, 1)), batch_size=1)
    batch_flt_hook = ModelHook(model.model, hook=retrieve_float_hook, Gamma=[0],
                               grad_bound=((0, 0), (1, 1)), batch_size=64)
    args.batch_size = 1
    train_loader, test_loader = set_loader(args)

    noise_attack = Noise(model.model, args.devices[0], 16 / 255, mean=(0.1307,), std=(0.3081,))

    # Record all the weight matrix
    r = 1
    all_w = []
    layers = []
    activations = []
    cur_layers = []
    for name, module in model.model.named_modules():
        if type(module) == torch.nn.Linear:
            cur_weight = to_numpy(module.weight)
            all_w.append(cur_weight)
        elif type(module) == torch.nn.BatchNorm1d:
            cur_weight = np.matmul(np.diag(to_numpy(module.weight)),
                                   np.diag(1 / to_numpy(torch.sqrt(module.running_var))))
            all_w.append(cur_weight)
        elif check_activation(module):
            all_w.append('A')

    for idx, (img, label) in enumerate(test_loader):
        model.model.eval()
        noise_img = noise_attack.attack(img, batch_size=64, device=args.devices[0])
        noise_img.require_grad = True

        a = model.model(noise_img)
        cur_batch_min = batch_flt_hook.retrieve_res(unpack)
        cur_local_min = single_flt_hk.retrieve_res(unpack)
        print(1)

        I = np.eye(args.input_size)
        act_counter = 0
        for w in all_w:
            if w != 'A':
                I = np.matmul(w, I)
            else:
                I = np.matmul(np.diag(cur_local_min[act_counter][0][1]), I)
                act_counter += 1

        r_1 = 1
        for layer in layers:
            for s, p in zip((1, 0, 1), (-1, 0, 1)):
                multiplier[b['layers.3'][0] == p] = s
            flt = np.linalg.svd(np.matmul(np.diag(multiplier), w_st))
            r_1 *= np.linalg.norm(layer[0][1].weight.cpu().detach().numpy(), 2)

        torch.autograd.functional.jacobian(model.model, noise_img[0])
        R = 1

        # for name, module in model.model.named_modules():
        #     if type(module) in [torch.nn.Conv2d, torch.nn.Linear]:
        #         cur_layers.append([name, module])
        #     elif check_activation(module):
        #         layers.append(cur_layers)
        #         activations.append([name, module])
        #         cur_layers = []
        # layers.append(cur_layers)

        """
        use norm to bound Lipschitz chonstant

        """
        output = torch.autograd.functional.jacobian(model.model, noise_img[0])
        w_st = layers[0][0][1].weight.detach().cpu().numpy()
        multiplier = np.ones(w_st.shape[0])
        for s, p in zip((1, 0, 1), (-1, 0, 1)):
            multiplier[b['layers.3'][0] == p] = s
        flt = np.linalg.svd(np.matmul(np.diag(multiplier), w_st))
        print(1)

        # storage = {}
        # noise_img = noise_attack.attack(img, batch_size=64, device=args.devices[0])
        # noise_img = to_device(args.devices[0], noise_img)[0]
        # for i in range(len(noise_img)):
        #     model.model(noise_img[i])
        #     input_hook.calculate(unpack, storage=storage)
        # pattern = {}
        # batch_p = retrieve_float_neurons(storage, [0])

        wm = [layers[i][0][1].weight for i in range(len(layers))]
        for mt_idx, ins_p in enumerate(batch_p[:, 0, :]):
            w_i = []
            for max_l, p in zip([1, 0, 1], (-1, 0, 1)):
                w_i.append(max_l * svd(wm[mt_idx][ins_p == p].cpu().detach().numpy())[1].max())

            print(1)

        img = img.cuda()
        img.requires_grad = True
        output = model.model(img)
        cost = model.loss_function(output, label.cuda())
        grad = torch.autograd.grad(cost, img, create_graph=True)

        for i in range(10):
            grad = torch.autograd.grad([output[:, i], output[:, 1]], img, create_graph=True)[0].cpu().detach().numpy()

        print(1)
    # for param in model.model.parameters():
    #     param.requires_grad = True
    # t = time.time()
    # for idx, (img, label) in enumerate(test_loader):
    #     region_diff, x_pattern = estimate_patters(model, img, sample_size=25, epsilon=0.2)
    #     layer_id = 0
    #     R = 1
    #     Ls = 1
    #     for layer in model.model.features:
    #         if type(layer) == torch.nn.Conv2d:
    #             r = region_diff[0].sum() / x_pattern[0].sum()
    #             R *= (1 + r)
    #             layer_id += 1
    #
    #             weight = layer.weight.permute([1, 0, 2, 3]).reshape([layer.weight.shape[1], -1]).cpu().detach().numpy()
    #             Ls *= svd(weight)[1][0] * 3
    #
    #     for layer in model.model.classifier:
    #         if type(layer) == torch.nn.Linear:
    #             weight = layer.weight.cpu().detach().numpy()
    #             EW1 = layer.weight[region_diff[layer_id]].cpu().detach().numpy()
    #             L = layer.weight[x_pattern[layer_id] == 0].cpu().detach().numpy()
    #             R *= (1 + svd(EW1)[1][0] / svd(L)[1][0])
    #
    #             Ls *= svd(weight)[1][0]
    #
    #             layer_id += 1
    #             if layer_id == len(x_pattern):
    #                 break
    #
    #     weight = model.model.classifier[-1].weight.cpu().detach().numpy()
    #     Ls *= svd(weight)[1][0]
    #
    #     img = img.cuda()
    #     img.requires_grad = True
    #     output = model.model(img)
    #     cost = model.loss_function(output, label.cuda())
    #     grad_mat = []
    #
    #     for i in range(10):
    #         grad = torch.autograd.grad([output[:, i], output[:, 1]], img, create_graph=True)[0].cpu().detach().numpy()
    #         grad_mat += [grad.flatten()]
    #     grad_mat = np.array(grad_mat)
    #     singular = svd(grad_mat)[1][0]
    #
    #     pre = model.model(img)[0]
    #     margin = (pre.sort()[0][-1] - pre.sort()[0][-2]).cpu().detach().numpy()
    #     ee = np.sqrt(margin / 3 / 32 / 32 / (singular * R))
    #     atk = FGSM(model.model, mean, std, ee)
    #     # atk = PGD(model.model, mean, std, ee)
    #     x = atk.attack(img, label.cuda())
    #
    #     # R_local += [R]
    #     # singular_local += [singular]
    #     # margin_local += [math.sqrt(margin / (32 * 32 * 3) / 172.39)]
    #     pre_adv = torch.argmax(model.model(x))
    #     if pre_adv.detach().cpu() == torch.argmax(pre).detach().cpu():
    #         safe += 1
    #     total += 1
    #     print(time.time() - t)
    #     t = time.time()
    #     if idx > 1000:
    #         break
    # print(1)
    # print(safe)
