from core.prune import *
from models.base_model import *
from settings.test_setting import *

if __name__ == '__main__':
    argv = ['--exp_id', 'l_0.00_b_0.1_e_0.8', '--batch_size', '128', '--net', 'vgg16', '--dataset', 'cifar100']
    res = []
    args = set_up_testing('normal', argv)
    model = BaseModel(args)
    model.load_model(args.model_dir)
    model.model.eval()
    _, test_loader = set_loader(args)

    ap_hook = ModelHook(model.model, retrieve_input_hook)
    metric = MetricLogger()
    storage = []
    net_same_all = []
    for idx, (images, labels) in enumerate(test_loader):
        images, labels = to_device(args.devices[0], images, labels)
        pre_ori = model.model(images)
        top1, top5 = accuracy(pre_ori, labels)
        unpacked = ap_hook.retrieve_res(unpack2)
        metric.update(top1_avd=(top1, args.batch_size), top5_adv=(top5, args.batch_size))
        if idx == 0:
            net_same_all = find_dead_neuron(unpacked, [0])
        else:
            net_same_all = compute_mean(net_same_all, find_dead_neuron(unpacked, [0]), idx)

    net_same_all.insert(0, [])
    model.model = model.model.cpu()
    block_counter = 1
    remove_conv = []
    remove_linear = []
    new_model = []
    new_size = []
    for n, m in model.named_modules():
        if type(m) in [LinearBlock, ConvBlock, BasicBlock, BottleNeck]:
            cur_block_ps = net_same_all[block_counter]
            pre_block_ps = net_same_all[block_counter - 1]
            m, shape = prune_block(m, cur_block_ps, pre_block_ps, 0.95)
            new_model += [m]
            new_size += [shape]
            block_counter += 1
        if type(m) == nn.MaxPool2d:
            new_size += ['M']
    wt = model.state_dict()
    args.config = new_size
    model.model = build_model(args)
    model.load_weights(wt)
    # metric2 = MetricLogger()
    # for idx, (images, labels) in enumerate(test_loader):
    #     images, labels = to_device(args.devices[0], images, labels)
    #     pre_ori = model.model(images)
    #     top1, top5 = accuracy(pre_ori, labels)
    #     unpacked = ap_hook.retrieve_res(unpack2)
    #     metric2.update(top1_avd=(top1, args.batch_size), top5_adv=(top5, args.batch_size))
    # print(1)
