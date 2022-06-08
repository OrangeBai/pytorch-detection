from attack.attack import *


class VANILA(Attack):
    def __init__(self, model, *args, **kwargs):
        super(VANILA, self).__init__("VANILA", model)

    def attack(self, images, labels=None, device=None):
        r"""
        Overridden.
        """

        return images


