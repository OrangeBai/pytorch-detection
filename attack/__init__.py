from attack.fgsm import FGSM
from attack.pgd import PGD
from attack.vanila import *
from attack.noise import *
from attack.ffgsm import FFGSM
from attack.Lip import LipAttack
# from .pgddlr import PGDDLR
# from .rfgsm import RFGSM
# from .apgd import APGD
# from .bim import BIM
# from .cw import CW
# from .tpgd import TPGD
# from .mifgsm import MIFGSM
# from .square import Square
# from .fab import FAB
# from .vanila import VANILA


def set_attack(model, name, device, *args, **kwargs):
    if name.upper() == 'Vanila':
        attack = VANILA(model, device, *args, **kwargs)
    elif name.upper() == 'FGSM':
        attack = FGSM(model, device, *args, **kwargs)
    elif name.upper() == 'PGD':
        attack = PGD(model, device, *args, **kwargs)
    elif name.upper() == 'FFGSM':
        attack = FFGSM(model, device, *args, **kwargs)
    elif name.upper() == 'LIP':
        attack = LipAttack(model, device, *args, **kwargs)
#     elif name == 'BIM':
#         attack = BIM(model, mean, std, eps=8 / 255, alpha=2 / 255, steps=7)
#     elif name == 'CW':
#         attack = CW(model, mean, std, c=1, kappa=0, steps=500, lr=0.01)
#     elif name == 'RFGSM':
#         attack = RFGSM(model, mean, std, eps=8 / 255, alpha=10 / 255, steps=1)
#     elif name == 'TPGD':
#         attack = TPGD(model, mean, std, eps=8 / 255, alpha=2 / 255, steps=7)
#     elif name == 'MIFGSM':
#         attack = MIFGSM(model, mean, std, eps=8 / 255, decay=1.0, steps=5)
#     elif name == 'APGD':
#         attack = APGD(model, mean, std, eps=8 / 255, steps=10)
#     elif name == 'FAB':
#         attack = FAB(model, mean, std, eps=8 / 255)
#     elif name == 'Square':
#         attack = Square(model, mean, std, eps=8 / 255)
#     elif name == 'PGDDLR':
#         attack = PGDDLR(model, mean, std, eps=8 / 255, alpha=2 / 255, steps=7)
    else:
        raise ModuleNotFoundError('Cannot find module {0}'.format(name))
    return attack
