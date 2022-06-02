from models import *
from settings.train_settings import *
import logging


if __name__ == '__main__':
    args = set_up_training()

    train_loader, test_loader = set_loader(args)

    model = BaseModel(args)
    logging.info(model.warmup(InfiniteLoader(train_loader)))
    inf_loader = InfiniteLoader(train_loader)