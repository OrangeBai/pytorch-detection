from attacks.attack import *
from core.utils import *
import numpy as np



class PGDDLR(Attack):
    def __init__(self, model, mean, std, eps=0.3, alpha=2 / 255, steps=40, restart=2):
        super(PGDDLR, self).__init__("FGSM", model, mean, std)
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.restart = restart

    def attack(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()
        images = self._reverse_norm(images)

        max_loss = torch.zeros(labels.shape[0]).cuda()
        max_delta = torch.zeros_like(images).cuda()

        adv_images = images.clone().detach()

        for zz in range(self.restart):
            # Starting at a uniformly random point
            delta = torch.zeros_like(images).cuda()
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

            for i in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.model(adv_images)

                cost = -self.dlr_loss(outputs, labels).sum()

                grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

                adv_images = adv_images.detach() - self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
                delta = adv_images - images
            all_loss = F.cross_entropy(self.model(images + delta), labels, reduction='none').detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

        adv_images = self._norm(adv_images)
        return adv_images

    @staticmethod
    def dlr_loss(x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (
                x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
