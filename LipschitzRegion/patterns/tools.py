import numpy as np
import torch
import os


def reshape_pattern(post_activation, batch_size):
    all_post_act = {idx: [] for idx in range(batch_size)}
    for idx, layer_post in post_activation.items():
        for item_idx, item_post in enumerate(layer_post):
            all_post_act[item_idx] += [item_post.flatten()]
    return np.array([item for item in all_post_act.values()])


def get_layer_pattern(post_activation, Gamma):
    """
    Get the activation pattern for current layer
    :param post_activation: The pre-activation values for the layer
    :param Gamma: name of activation
    :return: a string that recording the activation pattern for each neuron
    """
    res = np.zeros(post_activation.shape)
    for idx in range(len(Gamma) - 1):
        c = np.all([post_activation > Gamma[idx], post_activation < Gamma[idx + 1]], axis=0)
        res[c] = idx

    return res


def get_batch_pattern(post_activation, Gamma):
    batch_pattern = {}
    for key, val in post_activation.items():
        layer_pattern = np.zeros(val.shape)
        for gamma in Gamma:
            layer_pattern += val > gamma
        batch_pattern[key] = layer_pattern.reshape((layer_pattern.shape[0], -1))
    return batch_pattern


def get_diff(pattern_1, pattern_2):
    diff = []
    for layer_p1, layer_p2 in zip(pattern_1, pattern_2):
        diff += [layer_p1 != layer_p2]
    return diff


# def back_propagation()

def get_pattern_str(pattern):
    """
    Given a list of patterns, convert the pattern into string format
    :param pattern:
    :return:
    """
    str_pattern = [[str(int(x)) for x in list(single_pattern)] for single_pattern in pattern]
    s = list(map(lambda x: ''.join(x), str_pattern))
    return s


def add_noise(x, batch_size, epsilon=0.5):
    rand_var = torch.randn((batch_size,) + x.shape)
    flatten_var = torch.nn.Flatten()(rand_var)
    norm = torch.linalg.norm(flatten_var, dim=1, keepdim=True)
    standardized_noise = flatten_var.div(norm.expand_as(flatten_var))
    data_noise = torch.reshape(standardized_noise, (batch_size,) + x.shape)
    data_noise[0] = 0
    x = x + data_noise * epsilon
    return x


def add_hook(model, module_type, hook_type='pre'):
    post_activation = {}

    handles = []

    def set_forward_hook(module_name):
        def hook(l, input_var, output_var):
            if hook_type == 'pre':
                variable = input_var[0].cpu().detach().numpy()
            elif hook_type == 'post':
                variable = output_var.cpu().detach().numpy()
            else:
                raise ValueError('Variable %s not found' % hook_type)

            if module_name not in list(post_activation.keys()):
                post_activation[module_name] = variable
            else:
                post_activation[module_name] = np.row_stack([post_activation[module_name], variable])

        return hook

    layer = 0
    for name, module in model.model.named_modules():
        if type(module) == module_type:
            handles += [module.register_forward_hook(set_forward_hook(layer))]
            layer += 1

    return post_activation, handles


def cal_ps_matrix(post_activation, num_sample, Gamma):
    """
    Calculate the pattern similarity given given the post activations
    :param post_activation: The post activation values of a set
    :param num_sample: How many samples are calculated
    :param Gamma: The breakpoint set
    :return:
    """

    matrix = np.zeros([num_sample, num_sample])
    all_pattern = reformat_pattern(post_activation)
    for i in range(num_sample):
        for j in range(i, num_sample):
            matrix[i, j] += cal_ps_single(all_pattern[[i, j]], Gamma)
    return matrix


def reformat_pattern(post_activation):
    all_pattern = []
    for name, item in post_activation.items():
        all_pattern += [item]
    all_pattern = np.column_stack(all_pattern)
    return all_pattern


def cal_ps_pair_set(X, Y, Gamma):
    X, Y = reformat_pattern(X), reformat_pattern(Y)
    ps_xy = []
    for x, y in zip(X, Y):
        ps_xy += [cal_ps_single(np.array([x, y]), Gamma)]
    return ps_xy


def cal_ps_single(xy, Gamma):
    """
    Given a pair of pattern, calculate the pattern similarity of xy
    :param xy: [2, N] array
    :param Gamma: a set of breakpoints
    :return: number of same patterns
    """
    region = 0
    for idx in range(len(Gamma) - 1):
        region += np.all(np.row_stack([xy > Gamma[idx], xy < Gamma[idx + 1]]), axis=0).sum()
    return region


def evaluate_single_dataset(model, test_loader, gamma, num_sample):
    model.model.eval()

    post_activation, hooks = add_hook(model, 'Activation')

    for idx, (data, label) in enumerate(test_loader):
        data = data.cuda()
        pre = model.model(data)

    num_sample = 100
    matrix = cal_ps_matrix(post_activation, num_sample, gamma)

    res = []
    x = np.arange(0, 1, 0.02)
    for i in x:
        threshold = i * matrix[0, 0]
        res += [(np.sum((matrix > threshold)) - num_sample) / (num_sample * (num_sample - 1) / 2)]

    res = np.array(res)
    for hook in hooks:
        hook.remove()

    return x, res, matrix


class WeightRecorder:
    def __init__(self, model):
        self.model = model
        self.all_ratio = {}
        self.all_weights = {}
        self.all_grad = {}
        self.all_abs_grad = {}

        self.step_ratio = {}
        self.step_weights = {}
        self.step_grad = {}
        self.step_abs_grad = {}

        self.epoch_ratio = {}
        self.epoch_grad = {}
        self.epoch_abs_grad = {}
        self.counter = 0
        self.step_counter = 0

    def reset(self):
        self.epoch_ratio = {}
        self.epoch_grad = {}
        self.epoch_abs_grad = {}
        self.counter = 0

    def update_step_weight(self):
        self.counter += 1
        pre_weight = self.get_variable('weight')
        cur_grads = self.get_variable('grad')

        cur_ratio = self.get_ratio(pre_weight, cur_grads)

        # self.record_single_weight(pre_weight, cur_grads, cur_ratio)

        for key in cur_ratio:
            # Update the weights and grads for each step
            if key not in self.epoch_ratio.keys():
                self.epoch_ratio[key] = cur_ratio[key]
                self.epoch_grad[key] = cur_grads[key]
                self.epoch_abs_grad[key] = np.abs(cur_grads[key])
            else:
                self.epoch_ratio[key] += cur_ratio[key]
                self.epoch_grad[key] += cur_grads[key]
                self.epoch_abs_grad[key] += np.abs(cur_grads[key])

    def get_ratio(self, pre_weight, cur_grads):
        """
        get current ratio of the model
        :param pre_weight:
        :param cur_grads:
        :return:
        """
        cur_ratio = {}
        for name, module in self.model.named_modules():
            if 'Linear' in name:
                cur_ratio[name] = np.abs(cur_grads[name] / pre_weight[name])
                cur_ratio[name][np.where(np.abs(cur_ratio[name]) >= 5)] = 0
        return cur_ratio

    def get_variable(self, var_type):
        """
        Get current weights of the model
        :return:
        """
        cur_variable = {}
        for name, module in self.model.named_modules():
            if 'Linear' in name:
                if var_type == 'weight':
                    cur_variable[name] = module.weight.cpu().detach().numpy()
                elif var_type == 'grad':
                    cur_variable[name] = module.weight.grad.cpu().detach().numpy()
        return cur_variable

    def record_global(self, i):
        for key, val in self.epoch_ratio.items():
            self.epoch_ratio[key] = val / self.counter

        for key, val in self.epoch_abs_grad.items():
            self.epoch_abs_grad[key] = val / self.counter

        self.all_ratio[i] = self.epoch_ratio
        self.all_weights[i] = self.get_variable('weight')
        self.all_grad[i] = self.epoch_grad
        self.all_abs_grad[i] = self.epoch_abs_grad

        self.reset()

    """
    Below are used for recording step-wise weights and grads, no longer needed 
    """

    # def record_single_weight(self, pre_weight, cur_grads, cur_ratio):
    #     """
    #     Record the weights and grads during each step, abondon!!!
    #     :param pre_weight:
    #     :param cur_grads:
    #     :param cur_ratio:
    #     :return:
    #     """
    #     self.step_ratio[self.step_counter] = self.cal_mean_and_var(cur_ratio)
    #     self.step_weights[self.step_counter] = self.cal_mean_and_var(pre_weight)
    #     self.step_grad[self.step_counter] = self.cal_mean_and_var(cur_grads)
    #     self.step_abs_grad[self.step_counter] = self.cal_mean_and_var(cur_grads, abs_val=True)
    #
    #     self.step_counter += 1
    #
    # @staticmethod
    # def cal_mean_and_var(variable, abs_val=False):
    #     res = {}
    #     for name, val in variable.items():
    #         if abs_val:
    #             res[name] = np.array([np.abs(val).mean(), np.abs(val).var()])
    #         else:
    #             res[name] = np.array([val.mean(), val.var()])
    #     return res

    def save(self, model_dir):
        torch.save(self.all_ratio, os.path.join(model_dir, 'ratio.p'))
        torch.save(self.all_weights, os.path.join(model_dir, 'weights.p'))
        torch.save(self.all_grad, os.path.join(model_dir, 'grads.p'))
        torch.save(self.all_abs_grad, os.path.join(model_dir, 'abs_grads.p'))
        return


def record_test_result(model, test_dataloader):
    model.model.eval()

    post_activation, handles = add_hook(model, 'Activation', 'post')

    for idx, (test_data, test_label) in enumerate(test_dataloader):
        test_data = test_data.cuda()
        model.model(test_data)

    for name, val in post_activation.items():
        post_activation[name] = val[:256, :]

    for handle in handles:
        handle.remove()
    model.model.train()
    return post_activation


def estimate_patters(model, x, sample_size=20, epsilon=0.2):
    post_activation, handles = add_hook(model, torch.nn.ReLU)
    for idx in range(sample_size):
        perturbation = add_noise(x.squeeze(), 101, epsilon=epsilon)
        model.model(perturbation.cuda())
        if idx + 1 >= sample_size:
            break
    for handle in handles:
        handle.remove()
    batch_pattern = get_batch_pattern(post_activation, [0])

    region_diff = []
    for layer, patterns in batch_pattern.items():
        a = []
        diff = np.zeros(patterns.shape[1], dtype=bool)
        for idx, instance_pattern in enumerate(patterns[1:]):
            if idx % 100 == 0:
                a.append(diff.sum() / len(diff))
            else:
                diff += patterns[0] != instance_pattern
        region_diff += [diff]
    return region_diff, [value[0] for value in batch_pattern.values()]

#
# def cal_ps_diff_density(model, Gamma):
#     model.model.eval()
#     (test_1, test_2), (set_1, set_2) = get_test_loader(1, 2)
#
#     pre_1 = []
#     post_activation_1, handles = add_hook(model, 'Activation')
#     for idx, (data, label) in enumerate(test_1):
#         data = data.cuda()
#         pre_1 += [model.model(data).cpu().detach().numpy()]
#     pre_1 = np.row_stack(pre_1)
#     for handle in handles:
#         handle.remove()
#
#     pre_2 = []
#     post_activation_2, handles = add_hook(model, 'Activation')
#     for idx, (data, label) in enumerate(test_2):
#         data = data.cuda()
#         pre_2 += [model.model(data).cpu().detach().numpy()]
#     pre_2 = np.row_stack(pre_2)
#     for handle in handles:
#         handle.remove()
#
#     transition_density = []
#     # for idx, (x, y) in enumerate(zip(set_1.data, set_2.data)):
#     #     transition_density += [cal_transition_density(model, x, y, Gamma)]
#     #     if idx > 100:
#     #         break
#
#     a = cal_ps_pair_set(post_activation_1, post_activation_2, Gamma)
#     b = np.linalg.norm(pre_2 - pre_1, ord=2, axis=1)
#
#     model.model.train()
#     return a, b, transition_density
#
#
# def cal_transition_density(model, x, y, Gamma, num_of_points=512):
#     diff = (y - x) / (num_of_points - 1)
#     points = torch.stack([x + i * diff for i in range(num_of_points)])
#     post_activation, hooks = add_hook(model, 'Activation')
#     model.model(points.cuda())
#     for hook in hooks:
#         hook.remove()
#
#     reformatted_pattern = reformat_pattern(post_activation)
#     patterns = get_pattern(reformatted_pattern, Gamma)
#     return len(set(patterns))
#
#
# def get_Gamma(act):
#     if act == 'R' or act == 'ELU':
#         return [-99, 0, 99]
#     elif act == 'S':
#         return [-99, -1, 1, 99]
#     pass
