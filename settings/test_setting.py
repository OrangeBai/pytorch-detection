import yaml
from settings.base_funs import *


def set_up_testing():
    parser = get_args()
    args = parser.parse_args()

    parser.add_argument('--model_dir', default=name(args, False))
    with open(os.path.join(parser.parse_args().model_dir, 'args.yaml'), 'r') as file:
        args_dict = yaml.load(file, Loader=yaml.FullLoader)

    for key, val in args_dict.items():
        if key not in list(vars(parser.parse_args()).keys()):
            parser.add_argument('--' + key, default=val)

    return parser.parse_args()


if __name__ == '__main__':
    set_up_testing()
