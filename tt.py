from scipy.linalg import svd
from settings.test_setting import set_up_testing
from models.base_model import *
from core.pattern import *
from dataloader.base import *
from core.utils import *
from attack import *

if __name__ == '__main__':
    args = set_up_testing()
    model = BaseModel(args)

    model.load_model(args.model_dir)
    pattern_hook = ModelHook(model.model, hook=retrieve_pattern, Gamma=[0])
    input_hook = ModelHook(model.model, hook=retrieve_input_hook)
    args.batch_size = 1
    train_loader, test_loader = set_loader(args)

    noise_attack = Noise(model.model, args.devices[0], 4 / 255, mean=(0.1307,), std=(0.3081,))

    for idx, (img, label) in enumerate(test_loader):
        noise_img = noise_attack.attack(img, batch_size=64, device=args.devices[0])
        noise_img = to_device(args.devices[0], noise_img)[0]
        ll = model.model(noise_img)
        print(1)
