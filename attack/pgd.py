from attack.attack import *
from torch.nn import functional as F


class PGD(Attack):
    def __init__(self, model, device, *args, **kwargs):
        super(PGD, self).__init__(model, device, *args, **kwargs)
        self.eps = kwargs['eps'] if 'eps' in kwargs.keys() else 8 / 255
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs.keys() else 2 / 255
        self.steps = kwargs['steps'] if 'steps' in kwargs.keys() else 7
        self.restarts = kwargs['restarts'] if 'restarts' in kwargs.keys() else 1

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
                cost = loss_fn(outputs, labels)
                grad = torch.autograd.grad(cost, delta, retain_graph=False, create_graph=False)[0]
                d = delta
                g = grad
                if self.ord == 'inf':
                    d = torch.clamp(d + self.alpha * torch.sign(g), -self.eps, self.eps)  # bounds from epsilon  # bounds from image
                else:
                    grad_norm = grad.view(grad.shape[0], -1).norm(2, dim=-1, keepdim=True)
                    grad_norm = grad_norm.view(grad_norm.shape[0], grad_norm.shape[1], 1, 1)
                    d = d + self.alpha * grad / (grad_norm + 1e-8)
                    mask = d.view(d.shape[0], -1).norm(2, dim=1) <= self.eps
                    #
                    scaling_factor = d.view(d.shape[0], -1).norm(2, dim=-1) + 1e-8
                    scaling_factor[mask] = self.eps
                    #
                    d = d * self.eps / (scaling_factor.view(-1, 1, 1, 1))

                d = torch.clamp(d, 0 - images, 1 - images)
                delta.data = d

            all_loss = F.cross_entropy(self.model(images + delta), labels, reduction='none').detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

        adv_image = self._norm(images + max_delta)
        return adv_image
