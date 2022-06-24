from functools import reduce
from convex_adversarial import robust_loss
from core.pattern import *


def record_blocks(model):
    """
    Record all the weights of given model
    @param model: DNN or CNN model
    @return: model weights
    """

    blocks = []
    block_types = []
    for name, module in model.named_modules():
        if type(module) == LinearBlock:
            blocks.append(record_linear_weights(module))
            block_types.append('linear')
        elif type(module) == ConvBlock:
            blocks.append(record_conv_weights(module))
            block_types.append('conv')
        # TODO ResBlock
        else:
            continue
    return blocks, block_types


def record_linear_weights(block):
    """
    Record Weights of Linear Blocks
    @param block: A LinearBlock module
    @return: the weights of linear layer and the batch norm
    """
    block_weights = []
    cur_weights = []
    for name, module in block.named_modules():
        if type(module) == nn.Linear:
            cur_weights.append(to_numpy(module.weight))
        elif type(module) == nn.BatchNorm1d:
            cur_weights.append(np.diag(to_numpy(module.weight) / to_numpy(torch.sqrt(module.running_var))))
        elif check_activation(module):
            cur_weights.reverse()
            block_weights.append(reduce(np.dot, cur_weights))
            cur_weights = []
    return block_weights


def record_conv_weights(block):
    block_weights = []
    for name, module in block.named_modules():
        if type(module) == nn.Conv2d:
            block_weights.append(to_numpy(module.weight))
        elif type(module) == nn.BatchNorm2d:
            block_weights.append(to_numpy(module.weight) / to_numpy(torch.sqrt(module.running_var)))
    return block_weights


def compute_jac(temp_pt, block_weights, block_types, batch_size):
    all_w = {i: [] for i in range(batch_size)}
    for pattern, weights, block_type in zip(temp_pt[:-1], block_weights[:-1], block_types[:-1]):
        for w, (k, p) in zip(weights, pattern.items()):
            for j in range(len(p)):
                all_w[j] += [w, np.diag(p[j][1])]
    [v.reverse() for v in all_w.values()]
    return [reduce(np.matmul, instance_w) for instance_w in all_w.values()]


def amplify_ratio(local_lb, local_ub, block_weight, block_type):
    if block_type == 'linear':
        return linear_amplify(local_lb, local_ub, block_weight)
    elif block_type == 'conv':
        return conv_amplify(local_lb, local_ub, block_weight)


def linear_amplify(local_lb, local_ub, weight):
    float_neuron = local_lb[0] != local_ub[0]
    weight_2__norm = np.linalg.norm(weight[0], ord=2, axis=1)
    batch_float = np.array(local_ub[0]) * float_neuron
    batch_fixed = np.array(local_ub[0]) * (1 - float_neuron)
    batch_float_norm = np.linalg.norm(batch_float * weight_2__norm, ord=2, axis=1)
    batch_fixed_norm = np.linalg.norm(batch_fixed * weight_2__norm, ord=2, axis=1)
    return 1 + batch_float_norm / batch_fixed_norm

    # diag1 = np.apply_along_axis(np.diag, 1, np.array(float_neuron[0]) * np.array(local_pattern[0]))
    # fixed_matrix = torch.bmm(torch.tensor(diag1, dtype=float), torch.tensor(weight).repeat(len(diag1), 1, 1))
    # float_matrix = np.matmul(np.diag(1 - float_neuron), np.diag(local_ub), weight)
    # r += [1 + svd(float_matrix)[1][0] / svd(fixed_matrix)[1][0]]


def conv_amplify(local_pattern, local_ub, weight):
    if len(weight) > 1:
        single_integral = np.matmul(np.array(local_pattern[0] + local_ub[0]).sum(axis=(2, 3)) / 2, np.diag(weight[1])).sum(
            axis=1)
        region_integral = np.matmul(np.array(local_ub[0]).sum(axis=(2, 3)), np.diag(weight[1])).sum(axis=1)
    else:
        single_integral = (np.array(local_pattern[0] + local_ub[0]).sum(axis=(2, 3)) / 2).sum(axis=1)
        region_integral = np.array(local_ub[0]).sum(axis=(2, 3)).sum(axis=1)
    return region_integral / single_integral


def estimate_lip(args, model, images, sample_size):
    model.eval()
    mean, std = set_mean_sed(args)
    noise_attack = Noise(model, None, args.noise_eps, mean=mean, std=std)
    float_hook = ModelHook(model, set_pattern_hook, Gamma=set_gamma(args.activation))
    noised_sample = noise_attack.attack(images, sample_size, args.devices[0])
    noised_sample = [noised_sample[i:i + 512] for i in range(0, len(noised_sample), 512)]
    for n in noised_sample:
        model(to_device(args.devices[0], n)[0])
    region_lbs, region_ubs = float_hook.retrieve_res(retrieve_lb_ub, sample_size=sample_size,
                                                     grad_bound=set_lb_ub(args.activation))
    float_hook.remove()
    block_weights, block_types = record_blocks(model)
    ratio = np.ones(len(images))
    for block_lb, block_ub, block_weight, block_type in zip(region_lbs, region_ubs, block_weights, block_types):
        region_lbs, block_ub = pattern_to_bound([0, 1], block_lb, block_ub)
        ratio *= amplify_ratio(region_lbs, block_ub, block_weight, block_type)
    model.train()
    return ratio

# def compute_float(cur_input, noise_input, eps, gamma):
#     for cur_block, noise_block in zip(cur_input, noise_input):
#         for cur_layer, noise_layer in zip(cur_block, noise_block):
#             diff = (cur_layer - noise_layer) * eps


def pattern_to_bound(bound, *args):
    res = []
    for arg in args:
        float_p = np.array(arg, dtype=float)
        for i in range(len(bound)):
            float_p[float_p == i] = bound[i]
        res += [float_p]
    return res



# def linear_amplify(pattern, weight):
#     r = []
#     for w, (k, p) in zip(weight, pattern.items()):
#         for i in range(len(p)):
#             fixed_matrix = np.matmul(np.diag(p[i]), w)
#             float_matrix = np.matmul(np.diag(1 - p[i]), w)
#             r += [1 + svd(float_matrix)[1][0] / svd(fixed_matrix)[1][0]]
#     return np.array(r)
