from torch.nn.functional import one_hot

from core.lip import *
from core.utils import *
from engine import *
from models.blocks import FloatNet, DualNet


class GenTrainer(AdvTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.num_flt_est = args.num_flt_est
        self.est_lip = 0
        self.flt_net = FloatNet(self.model)
        self.dual_net = DualNet(self.model, set_gamma(args.activation))
        # self.attacks = self.set_attack