from torch.nn.functional import one_hot

from attack.attack import *
from core.utils import *


class LipAttack(Attack):
    def __init__(self, model, device, *args, **kwargs):
        super(LipAttack, self).__init__(model, device, *args, **kwargs)
        self.ord = kwargs['ord']

    def attack(self, images, labels, random_start=True, device=None):
        images = to_device(device, images.clone().detach())[0]
        images = self._reverse_norm(images)  # from normalized to (0,1)

        images.requires_grad = True
        outputs = self.model(images)
        flags = 1 - one_hot(labels, num_classes=outputs.shape[1]).type(torch.float).cuda()
        if self.ord == 'l2':
            cost = (outputs * flags).norm(p=2, dim=-1).mean()
        else:
            cost = (outputs * flags).norm(p=1, dim=-1).mean()

        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]

        if self.ord == 'l2':
            perturbation = grad / grad.norm(p=2, dim=(1, 2, 3)).view(len(grad), 1, 1, 1) * 0.0001
        else:
            perturbation = grad.sign() * 0.0001
        return perturbation
