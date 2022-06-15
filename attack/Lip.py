from attack.attack import *
from core.utils import *


class LipAttack(Attack):
    def __init__(self, model, device, *args, **kwargs):
        super(LipAttack, self).__init__(model, device, *args, **kwargs)
        self.eps = kwargs['eps'] if 'eps' in kwargs.keys() else 1 / 255

    def attack(self, images, labels, random_start=True, device=None):
        images = to_device(device, images.clone().detach())[0]
        labels = to_device(device, labels.clone().detach())[0]
        images = self._reverse_norm(images)  # from normalized to (0,1)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)  # from (0, 1) to normalized, and forward the emodel
        cost = outputs.norm(p=2, dim=1).mean()

        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        perturbation = grad / grad.norm(p=2, dim=(1, 2, 3)).view(128, 1, 1, 1) * 0.1
        # adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        #
        # adv_images = self._norm(adv_images)  # from (0,1) to normalized
        return perturbation
