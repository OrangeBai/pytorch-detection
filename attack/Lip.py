from attack.attack import *
from core.utils import *
from torch.nn.functional import one_hot


class LipAttack(Attack):
    def __init__(self, model, device, *args, **kwargs):
        super(LipAttack, self).__init__(model, device, *args, **kwargs)
        self.ord = kwargs['ord']

    def attack(self, images, labels, random_start=True, device=None):
        images = to_device(device, images.clone().detach())[0]
        images = self._reverse_norm(images)  # from normalized to (0,1)


        images.requires_grad = True
        outputs = self.model(images)
        fake_label = torch.randint(0, 10, (len(images),)).cuda()
        if self.ord == 'l2':
            cost = outputs.norm(p=2)
        else:
            bool_mat = one_hot(fake_label, num_classes=10).type(torch.float)
            cost = (outputs * bool_mat).norm(p=1)

        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]

        if self.ord == 'l2':
            perturbation = grad / grad.norm(p=2, dim=(1, 2, 3)).view(len(grad), 1, 1, 1) * 0.0001
            local_lip = (self.model(images + perturbation) - outputs) * 10000
        else:
            perturbation = grad.sign() * 0.0001
            local_lip = (self.model(images + perturbation) - outputs) * 10000
        return local_lip
