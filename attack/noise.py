import torch

from attack.attack import *
from core.utils import *


class Noise(Attack):
    def __init__(self, model, device, eps, *args, **kwargs):
        super(Noise, self).__init__(model, device, *args, **kwargs)
        self.eps = eps

    def attack(self, images, batch_size=128, device=None):
        shape = images.shape
        noise = self.eps * torch.sign(torch.randn((shape[0] * batch_size,) + shape[1:])).cuda()
        noise[0: shape[0] * batch_size : batch_size] = 0
        # images = images.cpu()
        images = self._reverse_norm(images)

        for i in range(shape[0]):
            noise[i * batch_size: (i + 1) * batch_size] += images[i]

        noisy_image = torch.clamp(noise, min=0, max=1)
        # s = noisy_image.shape
        # noisy_image = torch.reshape(noisy_image, (s[0] * s[1],) + s[2:])
        return self._norm(noisy_image)
