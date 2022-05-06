from scipy.linalg import svd
from settings.test_setting import set_up_testing
from models.base_model import *
from core.pattern import *
from dataloader.base import *
from core.utils import *

if __name__ == '__main__':
    args = set_up_testing()
    model = BaseModel(args)

    model.load_model(args.model_dir)
    pattern_hook = ModelHook(model.model, hook=retrieve_pattern, Gamma=[0])
    args.batch_size = 1
    train_loader, test_loader = set_loader(args)

    layers = []
    activations = []
    cur_layers = []

    for name, module in model.model.named_modules():
        if type(module) in [torch.nn.Conv2d, torch.nn.Linear]:
            cur_layers.append([name, module])
        elif check_activation(module):
            layers.append(cur_layers)
            activations.append([name, module])
            cur_layers = []

    for idx, (img, label) in enumerate(test_loader):
        img = add_noise(img, 101, epsilon=3/255)
        img = to_device(args.devices[0], img)[0]
        model.model(img)
        print(1)

        patterns = pattern_hook.calculate(retrieve_float_neurons)
        print(1)
    for param in model.model.parameters():
        param.requires_grad = True
    t = time.time()
    for idx, (img, label) in enumerate(test_loader):
        region_diff, x_pattern = estimate_patters(model, img, sample_size=25, epsilon=0.2)
        layer_id = 0
        R = 1
        Ls = 1
        for layer in model.model.features:
            if type(layer) == torch.nn.Conv2d:
                r = region_diff[0].sum() / x_pattern[0].sum()
                R *= (1 + r)
                layer_id += 1

                weight = layer.weight.permute([1, 0, 2, 3]).reshape([layer.weight.shape[1], -1]).cpu().detach().numpy()
                Ls *= svd(weight)[1][0] * 3

        for layer in model.model.classifier:
            if type(layer) == torch.nn.Linear:
                weight = layer.weight.cpu().detach().numpy()
                EW1 = layer.weight[region_diff[layer_id]].cpu().detach().numpy()
                L = layer.weight[x_pattern[layer_id] == 0].cpu().detach().numpy()
                R *= (1 + svd(EW1)[1][0] / svd(L)[1][0])

                Ls *= svd(weight)[1][0]

                layer_id += 1
                if layer_id == len(x_pattern):
                    break

        weight = model.model.classifier[-1].weight.cpu().detach().numpy()
        Ls *= svd(weight)[1][0]

        img = img.cuda()
        img.requires_grad = True
        output = model.model(img)
        cost = model.loss_function(output, label.cuda())
        grad_mat = []

        for i in range(10):
            grad = torch.autograd.grad([output[:, i], output[:, 1]], img, create_graph=True)[0].cpu().detach().numpy()
            grad_mat += [grad.flatten()]
        grad_mat = np.array(grad_mat)
        singular = svd(grad_mat)[1][0]

        pre = model.model(img)[0]
        margin = (pre.sort()[0][-1] - pre.sort()[0][-2]).cpu().detach().numpy()
        ee = np.sqrt(margin / 3 / 32 / 32 / (singular * R))
        atk = FGSM(model.model, mean, std, ee)
        # atk = PGD(model.model, mean, std, ee)
        x = atk.attack(img, label.cuda())

        # R_local += [R]
        # singular_local += [singular]
        # margin_local += [math.sqrt(margin / (32 * 32 * 3) / 172.39)]
        pre_adv = torch.argmax(model.model(x))
        if pre_adv.detach().cpu() == torch.argmax(pre).detach().cpu():
            safe += 1
        total += 1
        print(time.time() - t)
        t = time.time()
        if idx > 1000:
            break
    print(1)
    print(safe)
