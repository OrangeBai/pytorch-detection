from settings.base_funs import *
import yaml
import os


def set_up_training():
    parser = ArgParser().get_args()

    parser = model_dir(parser, True)
    parser = devices(parser)

    # save yaml file for test configuration
    args = parser.parse_args()
    args_dict = vars(args)
    json_file = os.path.join(args.model_dir, 'args.yaml')
    with open(json_file, 'w') as f:
        yaml.dump(args_dict, f)
    return parser.parse_args()
