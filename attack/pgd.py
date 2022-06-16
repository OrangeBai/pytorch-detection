from attack.attack import *
from torch.nn import functional as F


class PGD(Attack):
    def __init__(self, model, device, *args, **kwargs):
        super(PGD, self).__init__(model, device, *args, **kwargs)
        self.eps = kwargs['eps'] if 'eps' in kwargs.items() else 8 / 255
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs.items() else 2 / 255
        self.steps = kwargs['steps'] if 'steps' in kwargs.items() else 7
        self.restarts = kwargs['restarts'] if 'restarts' in kwargs.items() else False

    def attack(self, images, labels):
        images = to_device(self.device, images.clone().detach())[0]
        labels = to_device(self.device, labels.clone().detach())[0]
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
                d = torch.clamp(d + self.alpha * torch.sign(g), -self.eps, self.eps)  # bounds from epsilon
                d = torch.clamp(d, 0 - images, 1 - images)  # bounds from immage
                delta.data = d
                delta.grad.zero_()

            all_loss = F.cross_entropy(self.model(images + delta), labels, reduction='none').detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

        adv_image = self._norm(images + max_delta)
        return adv_image
