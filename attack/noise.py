from attack.base import *


class Noise(Attack):
    def __init__(self, model, device, eps, *args, **kwargs):
        super(Noise, self).__init__(model, device, *args, **kwargs)
        self.eps = eps

    def attack(self, images, labels, batch_size=128):
        shape = images.shape
        images = images.cuda()
        if len(images) > 1:
            raise ValueError('For noise attack, data len should be 1')
        noise = self.eps * torch.sign(torch.randn((batch_size,) + shape[1:])).cuda()
        noise[0] = 0
        images = self._reverse_norm(images)
        noise += images[0]

        noisy_image = torch.clamp(noise, min=0, max=1)
        return self._norm(noisy_image)
