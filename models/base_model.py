import importlib
import os
from collections import OrderedDict

from core.prune import *


class BaseModel(nn.Module):
    # TODO Record epoch info
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.set_up_kwargs = {'batch_norm': args.batch_norm, 'activation': args.activation}

    def forward(self, x):
        pass

    def save_model(self, path, name=None):
        if not name:
            model_path = os.path.join(path, 'weights.pth')
        else:
            model_path = os.path.join(path, 'weights_{}.pth'.format(name))
        torch.save(self.state_dict(), model_path)
        return

    def load_model(self, path, name=None):
        if not name:
            model_path = os.path.join(path, 'weights.pth')
        else:
            model_path = os.path.join(path, 'weights_{}.pth'.format(name))
        self.load_weights(torch.load(model_path))

        print('Loading model from {}'.format(model_path))
        return

    def load_weights(self, state_dict):
        new_dict = OrderedDict()
        for (k1, v1), (k2, v2) in zip(self.state_dict().items(), state_dict.items()):
            if v1.shape == v2.shape:
                new_dict[k1] = v2
            else:
                raise KeyError
        self.load_state_dict(new_dict)

    # def pruning_val(self, epoch, test_loader):
    #     start = time.time()
    #     self.model.eval()
    #     ap_hook = ModelHook(self.model, set_input_hook)
    #     metric = MetricLogger()
    #     storage = []
    #     net_same_all = []
    #     for idx, (images, labels) in enumerate(test_loader):
    #         images, labels = to_device(self.args.devices[0], images, labels)
    #         pre_ori = self.model(images)
    #         top1, top5 = accuracy(pre_ori, labels)
    #         unpacked = ap_hook.retrieve_res(unpack)
    #         metric.update(top1_avd=(top1, self.args.batch_size), top5_adv=(top5, self.args.batch_size))
    #         if idx == 0:
    #             net_same_all = find_dead_neuron(unpacked, [0])
    #         else:
    #             net_same_all = compute_mean(net_same_all, find_dead_neuron(unpacked, [0]), idx)
    #
    #     net_same_all.insert(0, [])
    #     self.model = self.model.cpu()
    #     block_counter = 1
    #     new_model = []
    #     new_size = []
    #     for n, m in self.model.named_modules():
    #         if type(m) in [LinearBlock, ConvBlock, BasicBlock, BottleNeck]:
    #             cur_block_ps = net_same_all[block_counter]
    #             pre_block_ps = net_same_all[block_counter - 1]
    #             m, shape = prune_block(m, cur_block_ps, pre_block_ps, 0.98)
    #             new_model += [m]
    #             new_size += [shape]
    #             block_counter += 1
    #         if type(m) == nn.MaxPool2d:
    #             new_size += ['M']
    #     wt = self.model.state_dict()
    #     self.args.config = new_size
    #     self.model = build_model(self.args)
    #     self.load_weights(wt)
    #
    #     self.model.train()
    #     self.model = self.model.cuda()
    #     self.optimizer.param_groups[0]['params'].clear()
    #     self.optimizer.param_groups[0]['params'].append(self.model.parameters())
    #     msg = self.val_logging(epoch) + '\ttime:{0:.4f}'.format(time.time() - start)
    #     self.logger.info(msg)
    #
    #     print(msg)
    #     print('Triming to {0}'.format(new_size))
    #     self.logger.info('Triming to {0}'.format(new_size))
    #     return


def build_model(args):
    """Import the module "model/[model_name]_model.py"."""
    model = None
    if args.model_type == 'dnn':
        model_file_name = "models." + args.model_type
        modules = importlib.import_module(model_file_name)
        model = modules.__dict__['DNN'](args)
    else:
        model_file_name = "models." + "net"
        modules = importlib.import_module(model_file_name)
        model = modules.set_model(args)
        # for name, cls in modules.__dict__.items():
        #     if name.lower() in args.net.lower():
        #         model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            model_file_name, args.net))
        exit(0)
    else:
        return to_device(args.devices[0], model)[0]
