from attacks.attack import *


class BIM(Attack):
    def __init__(self, model, mean, std, eps=4 / 255, alpha=1 / 255, steps=0):
        super(BIM, self).__init__("BIM", model, mean, std)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        else:
            self.steps = steps

    def attack(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()
        images = self._reverse_norm(images)

        loss = nn.CrossEntropyLoss()

        ori_images = images.clone().detach()

        for i in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)
            cost = -loss(outputs, labels)

            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]

            adv_images = images - self.alpha * grad.sign()
            # a = max(ori_images-eps, 0)
            a = torch.clamp(ori_images - self.eps, min=0)
            # b = max(adv_images, a) = max(adv_images, ori_images-eps, 0)
            b = (adv_images >= a).float() * adv_images \
                + (adv_images < a).float() * a
            # c = min(ori_images+eps, b) = min(ori_images+eps, max(adv_images, ori_images-eps, 0))
            c = (b > ori_images + self.eps).float() * (ori_images + self.eps) \
                + (b <= ori_images + self.eps).float() * b
            # images = max(1, c) = min(1, ori_images+eps, max(adv_images, ori_images-eps, 0))
            images = torch.clamp(c, min=0, max=1).detach()

        images = self._norm(images)
        return images
