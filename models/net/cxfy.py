from models.blocks import *


class CXFY(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_cls = args.num_cls