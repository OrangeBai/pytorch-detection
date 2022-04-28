from attacks.attack import *


class RFGSM(Attack):
    def __init__(self, model, mean, std, eps, alpha, steps=1):
        super(RFGSM, self).__init__("RFGSM", model, mean, std)
        self.epsilon = eps
        self.alpha = alpha
        self.steps = steps

    def attack(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()
        images = self._reverse_norm(images)
        loss = nn.CrossEntropyLoss()

        adv_images = images + self.alpha * torch.randn_like(images).sign()
        adv_images = clamp(adv_images, self.lower_limit, self.upper_limit).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            cost = loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images + (self.epsilon + self.alpha) * grad.sign()
            adv_images = torch.clamp(adv_images, 0, 1).detach()

        adv_images = self._norm(adv_images)
        return adv_images
