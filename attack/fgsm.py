from attack.attack import *
from core.utils import *


class FGSM(Attack):
    def __init__(self, model, device, *args, **kwargs):
        super(FGSM, self).__init__(model, device, *args, **kwargs)
        self.eps = kwargs['eps'] if 'eps' in kwargs.keys() else 8 / 255

    def attack(self, images, labels):
        images = to_device(self.device, images.clone().detach())[0]
        labels = to_device(self.device, labels.clone().detach())[0]
        images = self._reverse_norm(images)  # from normalized to (0,1)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)  # from (0, 1) to normalized, and forward the emodel
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        adv_images = self._norm(adv_images)  # from (0,1) to normalized
        return adv_images
