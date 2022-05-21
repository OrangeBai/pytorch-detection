from scipy.linalg import svd
from settings.APLip_settings import set_up_testing
from models.base_model import *
from core.pattern import *
from dataloader.base import *
from attack import *

if __name__ == '__main__':
    args = set_up_testing()
    model = BaseModel(args)
    model.load_model(args.model_dir)
    train_loader, test_loader = set_loader(args)
    # TODO refactor noise module
    noise_attack = Noise(model.model, args.devices[0], args.epsilon[0], mean=(0.5,), std=(1,))

    float_hook = ModelHook(model.model, hook=float_neuron_hook, Gamma=[0], sample_size=args.sample_size)
    ub_lb_hook = ModelHook(model.model, hook=lb_ub_hook, Gamma=[0],
                           grad_bound=[(0, 0), (1, 1)], sample_size=args.sample_size)
    # Record all the weight matrix
    r = 1
    avg = []
    t = time.time()
    for idx, (img, label) in enumerate(test_loader):
        model.model.eval()
        noise_img = noise_attack.attack(img, batch_size=args.sample_size, device=args.devices[0])
        noise_img.require_grad = True

        a = model.model(noise_img)
        temp_mt = ub_lb_hook.retrieve_res(unpack)




        eye_matrix = np.eye(args.input_size)
        act_counter = 0
        for w in all_w:
            if w != 'A':
                eye_matrix = np.matmul(w, eye_matrix)
                # pass
            else:
                eye_matrix = np.matmul(np.diag(temp_mt[act_counter][0][1]), eye_matrix)
                act_counter += 1
                pass
        local_lip = svd(eye_matrix)[1][0]

        r = 1
        linear_counter = 0
        cur_float = float_hook.retrieve_res(unpack)
        for layer in model.model.modules():
            if type(layer) == torch.nn.Linear:
                weight = layer.weight.cpu().detach().numpy()
                fix_weights = np.matmul(np.diag(cur_float[0].astype(float) * temp_mt[linear_counter][0][1]), weight)
                sigma_1 = svd(fix_weights)[1][0]
                float_weights = np.matmul(
                    np.diag((1 - cur_float[linear_counter].astype(float)) * temp_mt[linear_counter][0][1]),
                    weight)
                sigma_flt = svd(float_weights)[1][1]
                # sigma_div = svd(np.eye(weight.shape[0]) + np.matmul(float_weights, np.linalg.pinv(fix_weights)))[1][0]
                r *= (1 + sigma_flt / sigma_1)
                # r *= sigma_div
                linear_counter += 1
                if linear_counter >= len(temp_mt):
                    break

            if type(layer) == torch.nn.Conv2d:
                pass
                # weight = layer.weight.cpu().detach().numpy()
                # fix_weights = np.matmul(np.diag(cur_float[0].astype(float) * temp_mt[linear_counter][0][1]), weight)
                # sigma_1 = max(svd(fix_weights)[1][0], 1)
                # float_weights = np.matmul(np.diag((1 - cur_float[0].astype(float)) * temp_mt[linear_counter][0][1]),
                #                           weight)
                # sigma_flt = svd(float_weights)[1][0]
                # r *= (1 + sigma_flt / sigma_1)
                # linear_counter += 1
        est = svd(eye_matrix)[1][0] * r
        avg += [est]
        ub_lb_hook.reset()

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
