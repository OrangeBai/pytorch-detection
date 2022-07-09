from functools import reduce
from core.pattern import *
from models.blocks import LinearBlock, ConvBlock


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


def list_blocks(model):
    """
    Record all the weights of given model
    @param model: DNN or CNN model
    @return: model weights
    """

    blocks = []
    block_types = []
    for name, module in model.named_modules():
        if type(module) in [LinearBlock, ConvBlock]:
            blocks.append(module)
    return blocks


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


def amplify_ratio(block_flt, block_ub, block, block_input, g):
    if type(block) == LinearBlock:
        return linear_amplify(block_flt, block_ub, block, block_input, g)
    elif type(block) == ConvBlock:
        return conv_amplify(block_flt, block_ub, block, block_input, g)


def linear_amplify(local_lb, local_ub, weight):

    # float_neuron = local_lb[0] != local_ub[0]
    # weight_2__norm = np.linalg.norm(weight[0], ord=2, axis=1)
    # batch_float = np.array(local_ub[0]) * float_neuron
    # batch_fixed = np.array(local_ub[0]) * (1 - float_neuron)
    # batch_float_norm = np.linalg.norm(batch_float * weight_2__norm, ord=2, axis=1)
    # batch_fixed_norm = np.linalg.norm(batch_fixed * weight_2__norm, ord=2, axis=1)
    return 1 + batch_float_norm / batch_fixed_norm

    # diag1 = np.apply_along_axis(np.diag, 1, np.array(float_neuron[0]) * np.array(local_pattern[0]))
    # fixed_matrix = torch.bmm(torch.tensor(diag1, dtype=float), torch.tensor(weight).repeat(len(diag1), 1, 1))
    # float_matrix = np.matmul(np.diag(1 - float_neuron), np.diag(local_ub), weight)
    # r += [1 + svd(float_matrix)[1][0] / svd(fixed_matrix)[1][0]]


def conv_amplify(block_flt, block_ub, block, block_input, g):
    fixed_norm = (block(block_input) - block(block_input + g)).norm(p=2) * 1e4
    running_mean = block.BN.weight / torch.sqrt(block.BN.running_var)
    weight = block.Conv.weight
    weight = weight * running_mean.view(len(running_mean), 1, 1, 1)
    # power_iteration_conv_evl(block_input, block.Conv, 1000, None)
    ub = torch.tensor(block_ub[0], dtype=weight.dtype).cuda() * (1-torch.tensor(block_flt[0], dtype=weight.dtype).cuda())
    EPS = 1e-24
    u = torch.randn((1, *block_input.size()[1:])).cuda()
    v = u / (u + EPS)

    for i in range(1000):
        u1 = _conv2d(u, weight, None, stride=block.Conv.stride, padding=block.Conv.padding)
        u1 = u1 * ub
        u1_norm = u1.norm(2)
        v = u1 / (u1_norm + EPS)
        u_tmp = u

        v1 = _conv_trans2d(v, weight, stride=block.Conv.stride, padding=block.Conv.padding, output_padding=0)
        #  When the output size of conv_trans differs from the expected one.
        if v1.shape != u.shape:
            v1 = _conv_trans2d(v, weight, stride=block.Conv.stride, padding=block.Conv.padding, output_padding=1)
        v1_norm = v1.norm(2)
        u = v1 / (v1_norm + EPS)

        if (u - u_tmp).norm(2) < 1e-5:
            break

    out = (v * _conv2d(u, weight, None, stride=block.Conv.stride, padding=block.Conv.padding)).view(v.size()[0], -1).sum(1)[0]
    return 1 + out / fixed_norm


def _conv2d(x, w, b, stride=1, padding=0):
    return F.conv2d(x, w, bias=b, stride=(1,1), padding=(1,1))


def _conv_trans2d(x, w, stride=1, padding=0, output_padding=0):
    return F.conv_transpose2d(x, w, stride=stride, padding=padding, output_padding=output_padding)


def power_iteration_conv_evl(mu, layer, num_simulations, u=None):
    EPS = 1e-24
    output_padding = 0
    if u is None:
        u = torch.randn((1, *mu.size()[1:])).cuda()

    W = layer.weight
    if layer.bias is not None:
        b = torch.zeros_like(layer.bias)
    else:
        b = None
    for i in range(num_simulations):
        u1 = _conv2d(u, W, b, stride=layer.stride, padding=layer.padding)
        u1_norm = u1.norm(2)
        v = u1 / (u1_norm + EPS)
        u_tmp = u

        v1 = _conv_trans2d(v, W, stride=layer.stride, padding=layer.padding, output_padding=output_padding)
        #  When the output size of conv_trans differs from the expected one.
        if v1.shape != u.shape:
            output_padding = 1
            v1 = _conv_trans2d(v, W, stride=layer.stride, padding=layer.padding, output_padding=output_padding)
        v1_norm = v1.norm(2)
        u = v1 / (v1_norm + EPS)

        if (u - u_tmp).norm(2) < 1e-5 or (i + 1) == num_simulations:
            break

    out = (v * (_conv2d(u, W, b, stride=layer.stride, padding=layer.padding))).view(v.size()[0], -1).sum(1)[0]
    return out, u

# def estimate_lip(args, model, images, sample_size):
#     model.eval()
#     mean, std = set_mean_sed(args)
#     noise_attack = Noise(model, None, args.noise_eps, mean=mean, std=std)
#     float_hook = ModelHook(model, set_pattern_hook, Gamma=set_gamma(args.activation))
#     noised_sample = noise_attack.attack(images, sample_size, args.devices[0])
#     noised_sample = [noised_sample[i:i + 512] for i in range(0, len(noised_sample), 512)]
#     for n in noised_sample:
#         model(to_device(args.devices[0], n)[0])
#     region_lbs, region_ubs = float_hook.retrieve_res(retrieve_lb_ub, sample_size=sample_size,
#                                                      grad_bound=set_lb_ub(args.activation))
#     float_hook.remove()
#     block_weights, block_types = record_blocks(model)
#     ratio = np.ones(len(images))
#     for block_lb, block_ub, block_weight, block_type in zip(region_lbs, region_ubs, block_weights, block_types):
#         region_lbs, block_ub = pattern_to_bound([0, 1], block_lb, block_ub)
#         ratio *= amplify_ratio(region_lbs, block_ub, block_weight, block_type)
#     model.train()
#     return ratio

# def compute_float(cur_input, noise_input, eps, gamma):
#     for cur_block, noise_block in zip(cur_input, noise_input):
#         for cur_layer, noise_layer in zip(cur_block, noise_block):
#             diff = (cur_layer - noise_layer) * eps




# def linear_amplify(pattern, weight):
#     r = []
#     for w, (k, p) in zip(weight, pattern.items()):
#         for i in range(len(p)):
#             fixed_matrix = np.matmul(np.diag(p[i]), w)
#             float_matrix = np.matmul(np.diag(1 - p[i]), w)
#             r += [1 + svd(float_matrix)[1][0] / svd(fixed_matrix)[1][0]]
#     return np.array(r)
