import torch
from core.utils import *
from settings.cifar_settings import *
from models.mini.model import Model
import logging
from dataloaders.cifar import *

if __name__ == '__main__':
    args = set_up_training()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.model_dir, 'logger')
    )
    logger.info(args)

    model = Model(args, logger)
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.dataset)

    for epoch in range(args.num_epoch):
        for idx, (images, labels) in model.metrics.epoch_logger(train_loader):
            model.train_step(images, labels)
            if idx % args.print_every == 0:
                log_msg = model.metrics.logging(idx, len(train_loader), epoch, args.num_epoch)
                logger.info(log_msg)
                print(log_msg)
        model.lr_scheduler.step()
    model.save_model(args.model_dir, 'weights.p')


