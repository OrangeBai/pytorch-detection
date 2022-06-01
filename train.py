from models import *
from settings.train_settings import *
import logging
from core.pattern import *

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

    model = BaseModel(args)
    logging.info(model.warmup(train_loader))
    inf_loader = InfiniteLoader(train_loader)

    for cur_epoch in range(args.num_epoch):
        for cur_step in range(args.epoch_step):
            images, labels = next(inf_loader)
            if args.reg:
                model.train_step_min_reg(images, labels)
            else:
                model.train_step(images, labels)

            if cur_step % args.print_every == 0:
                log_msg = model.train_logging(cur_step, args.epoch_step, cur_epoch, args.num_epoch, inf_loader.metric)
                logging.info(log_msg)
                print(log_msg)
            model.lr_scheduler.step()

        log_msg = model.epoch_logging(cur_epoch, args.num_epoch, time_metrics=inf_loader.metric)
        inf_loader.reset()

        log_msg += model.validate_model(cur_epoch, test_loader) + '\n'
        logging.info(log_msg)
        print(log_msg)
    model.save_model(args.model_dir)
    model.save_result(args.model_dir)
