from models import *
from settings.train_settings import *
import logging

if __name__ == '__main__':
    args = set_up_training()

    train_loader, test_loader = set_loader(args)

    model = BaseModel(args)
    logging.info(model.warmup(InfiniteLoader(train_loader)))
    inf_loader = InfiniteLoader(train_loader)

    for cur_epoch in range(args.num_epoch):
        for cur_step in range(args.epoch_step):
            images, labels = next(inf_loader)

            if args.lmd != 0:
                model.train_step_min_reg(images, labels)
            else:
                model.train_step(images, labels)

            if cur_step % args.print_every == 0:
                model.train_logging(cur_step, args.epoch_step, cur_epoch, args.num_epoch, inf_loader.metric)

        model.epoch_logging(cur_epoch, args.num_epoch, time_metrics=inf_loader.metric)
        inf_loader.reset()

        model.validate_model(cur_epoch, test_loader)

    model.save_model(args.model_dir)
    model.save_result(args.model_dir)
