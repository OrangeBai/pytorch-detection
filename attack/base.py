from core.utils import *


class Attack(nn.Module):
    def __init__(self, model, device, *args, **kwargs):
        super().__init__()
        # TODO Temporary solution for multi-device, need modification

        mean = kwargs['mean'] if 'mean' in kwargs.keys() else [0, 0, 0]
        std = kwargs['std'] if 'std' in kwargs.keys() else [1, 1, 1]
        self.ord = kwargs['ord'] if 'ord' in kwargs.keys() else 'inf'
        self.device = device
        self.norm_layer = Normalize(mean=mean, std=std)

        self.model, self.mean, self.std = to_device(device,
                                                    nn.Sequential(self.norm_layer, model),
                                                    torch.tensor(mean).view(len(mean), 1, 1),
                                                    torch.tensor(std).view(len(std), 1, 1))
        # self.model = nn.Sequential(self.norm_layer, model).cuda()
        # self.mean = torch.tensor(mean).view(len(mean), 1, 1).cuda()
        # self.std = torch.tensor(std).view(len(mean), 1, 1).cuda()

        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)

    def _reverse_norm(self, x):
        return x * self.std + self.mean

    def _norm(self, x):
        return (x - self.mean) / self.std

    def attack(self, *args):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, images):
        # Broadcasting
        mean = self.mean.reshape(1, self.mean.shape[0], 1, 1)
        std = self.std.reshape(1, self.mean.shape[0], 1, 1)
        return (images - mean) / std
