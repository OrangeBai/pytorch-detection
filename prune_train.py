from settings.train_settings import *

if __name__ == '__main__':
    arg_parser = ArgParser(True)
    for file_path in arg_parser.files:
        arg_parser.modify_parser(file_path)

        train_model(arg_parser.get_args())



