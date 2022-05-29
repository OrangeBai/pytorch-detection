from settings.APLip_settings import set_up_testing
from models.base_model import *
from core.pattern import *
from dataloader.base import *
from attack import *
from Lip.utils import *
if __name__ == '__main__':
    args = set_up_testing()
    model = BaseModel(args)
    model.load_model(args.model_dir)
    train_loader, test_loader = set_loader(args)

    pdg = FGSM(model.model, args.devices[0], args.epsilon[0])