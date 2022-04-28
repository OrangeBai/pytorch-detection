from attacks.attack import *


class PGD(Attack):
    def __init__(self, model, mean, std, eps, alpha, steps=10, restarts=2):
        super(PGD, self).__init__("PGD", model, mean, std)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.restarts = restarts

    def attack(self, images, labels, random_start=True):
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()
        images = self._reverse_norm(images)
        loss_fn = nn.CrossEntropyLoss()

        max_loss = torch.zeros(labels.shape[0]).cuda()
        max_delta = torch.zeros_like(images).cuda()

        for zz in range(self.restarts):
            delta = torch.zeros_like(images).cuda()
            delta.uniform_(-self.eps, self.eps)
            delta.data = torch.clamp(delta, min=-self.eps, max=self.eps)
            delta.requires_grad = True
            for i in range(self.steps):
                outputs = self.model(images + delta)
                loss = loss_fn(outputs, labels)
                loss.backward()
                grad = delta.grad.detach()
                d = delta
                g = grad
                d = torch.clamp(d + self.alpha * torch.sign(g), -self.eps, self.eps)
                d = clamp(d, 0 - images, 1 - images)
                delta.data = d
                delta.grad.zero_()

            all_loss = F.cross_entropy(self.model(images + delta), labels, reduction='none').detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

        adv_image = self._norm(images + max_delta)
        return adv_image
