from attack.attack import *
from core.utils import *


class FGSM(Attack):
    def __init__(self, model, mean, std, eps):
        super(FGSM, self).__init__("FGSM", model, mean, std)
        self.eps = eps

    def attack(self, images, labels, random_start=True):
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()
        images = self._reverse_norm(images)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        adv_images = self._norm(adv_images)
        return adv_images
