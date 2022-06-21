from attack.attack import *
import torch.optim as optim
import torch


class CW(Attack):
    def __init__(self, model, mean, std, c=1e-4, kappa=0, steps=100, lr=0.01):
        super(CW, self).__init__("CW", model, mean, std)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    def attack(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()
        images = self._reverse_norm(images)
        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).cuda()
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get Adversarial Images
            adv_images = self.tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.model(adv_images)
            f_Loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c * f_Loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update Adversarial Images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early Stop when loss does not converge.
            if step % (self.steps // 10) == 0:
                if cost.item() > prev_cost:
                    return self._norm(best_adv_images)
                prev_cost = cost.item()

        best_adv_images = self._norm(best_adv_images)
        return best_adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x * 2 - 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].cuda()

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp(-(i - j), min=-self.kappa)
