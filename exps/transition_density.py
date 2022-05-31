import torch
from settings.test_setting import *
from models.base_model import *

if __name__ == '__main__':
    args = set_up_testing()
    model = BaseModel(args)
    model.load_model(args.model_dir)
    train_loader, test_loader = set_loader(args)
