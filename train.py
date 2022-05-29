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
    inf_loader = InfiniteLoader(train_loader)

    # att = get_attack(model.model, args.attack, args.devices[0], mean=dataloader.cifar.get_mean_std('cifar10')[0],
    #                  std=dataloader.cifar.get_mean_std('cifar10')[1])

    # pattern_hook = ModelHook(model, retrieve_pattern, Gamma=[0])
    for cur_epoch in range(args.num_epoch):
        for cur_step in range(args.epoch_step):
            images, labels = next(inf_loader)
            # adv_images = att.attack(images, labels, device=args.devices[0])
            if args.reg:
                model.train_step_min_reg(images, labels)
            else:
                model.train_step(images, labels)
            # cal = pattern_hook.calculate(retrieve_float_neurons)

            if cur_step % args.print_every == 0:
                log_msg = model.logging(cur_step, args.epoch_step, cur_epoch, args.num_epoch, inf_loader.metric)
                logging.info(log_msg)
                print(log_msg)

        # TODO refactor the validation part, add attack models
        log_msg = model.epoch_logging(cur_epoch, args.num_epoch, time_metrics=inf_loader.metric)
        inf_loader.reset()

        log_msg += model.validate_model(cur_epoch, test_loader) + '\n'
        logging.info(log_msg)
        print(log_msg)
    model.save_model(args.model_dir)
    model.save_result(args.model_dir)
