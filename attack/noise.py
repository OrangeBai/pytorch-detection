import torch

from attack.attack import *
from core.utils import *


class Noise(Attack):
    def __init__(self, model, device, eps, *args, **kwargs):
        super(Noise, self).__init__(model, device, *args, **kwargs)
        self.eps = eps

    def attack(self, images, batch_size=128, device=None):
        noise = torch.sign(torch.randn((batch_size,) + images.shape))
        noise[0] = 0
        images, noise = to_device(device, images, noise)
        images = self._reverse_norm(images)

        noisy_image = torch.clamp(images + self.eps * noise, min=0, max=1)
        # s = noisy_image.shape
        # noisy_image = torch.reshape(noisy_image, (s[0] * s[1],) + s[2:])
        return self._norm(noisy_image)
