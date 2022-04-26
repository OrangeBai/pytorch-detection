from attacks.attack import *


class FFGSM(Attack):
    def __init__(self, model, mean, std, eps=8 / 255, alpha=10 / 255):
        super(FFGSM, self).__init__("FFGSM", model, mean, std)
        self.eps = eps
        self.alpha = alpha

    def attack(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()
        images = self._reverse_norm(images)

        loss = nn.CrossEntropyLoss()

        delta = torch.zeros_like(images).cuda()
        delta.uniform_(-self.eps, self.eps)
        delta.data = clamp(delta, 0 - images, 1 - images)
        delta.requires_grad = True

        outputs = self.model(images + delta)
        cost = -loss(outputs, labels)

        cost.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta - self.alpha * torch.sign(grad), -self.eps, self.eps)
        delta.data = clamp(delta, 0 - images, 1-images)
        delta = delta.detach()
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        adv_images = self._norm(adv_images)
        return adv_images
