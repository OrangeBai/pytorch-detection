from attacks.attack import *


class TPGD(Attack):
    def __init__(self, model, mean, std, eps=8 / 255, alpha=2 / 255, steps=7):
        super(TPGD, self).__init__("TPGD", model, mean, std)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self._attack_mode = 'only_default'

    def attack(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().cuda()
        images = self._reverse_norm(images)
        logit_ori = self.model(images).detach()

        adv_images = images + 0.001 * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        loss = nn.KLDivLoss(reduction='sum')

        for i in range(self.steps):
            adv_images.requires_grad = True
            logit_adv = self.model(adv_images)

            cost = loss(F.log_softmax(logit_adv, dim=1),
                        F.softmax(logit_ori, dim=1))

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        adv_images = self._norm(adv_images)
        return adv_images
