from attacks.attack import *


class VANILA(Attack):
    def __init__(self, model):
        super(VANILA, self).__init__("VANILA", model)

    def attack(self, images, labels=None):
        r"""
        Overridden.
        """
        adv_images = images.clone().detach().cuda()

        return adv_images
