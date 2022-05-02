from models import *
from dataloader.base import *
from settings.cifar_settings import *
from attack import *
from core.utils import *
import logging

if __name__ == '__main__':
    args = set_up_training()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.model_dir, 'logger'))
    logger.info(args)

    train_loader, test_loader = set_loader(args)

    if args.train_type == 'epoch':
        # set total step, epoch step for epoch-wise training
        # set num of epoch for  step-wise training
        args.epoch_step = len(train_loader)
        args.total_step = args.epoch_step * args.num_epoch
    else:
        args.num_epoch = args.total_step // args.epoch_step

    model = BaseModel(args, logger)
    inf_loader = InfiniteLoader(train_loader)

    if args.attack is not None:
        att = get_attack(model.model, 'PGD', dataloader.cifar.get_mean_std('cifar10')[0],
                     dataloader.cifar.get_mean_std('cifar10')[1])

    for cur_epoch in range(args.num_epoch):
        for cur_step in range(args.epoch_step):
            images, labels = next(inf_loader)
            model.train_step(images, labels)

            if args.attack is not None:
                adv_images = att.attack(images, labels)

            if cur_step % args.print_every == 0:
                time_metrics = inf_loader.pack_metric()
                log_msg = model.logging(cur_step, args.epoch_step, cur_epoch, args.num_epoch, time_metrics)
                logging.info(log_msg)
                print(log_msg)

        # TODO refactor the validation part, add attack models
        log_msg = model.epoch_logging(cur_epoch, args.num_epoch, time_metrics=inf_loader.pack_metric(reset=True))
        logging.info(log_msg)
        print(log_msg)

        model.validate_model(cur_epoch, test_loader)
        print(log_msg)


