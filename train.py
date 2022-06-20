from core.engine import *
from settings.train_settings import *
import numpy as np


if __name__ == '__main__':
    arg_parser = ArgParser(True)
    for file_path in arg_parser.files:
        arg_parser.modify_parser(file_path)

        trainer = Trainer(arg_parser.get_args())
        trainer.train_model()




