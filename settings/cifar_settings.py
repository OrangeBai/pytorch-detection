import argparse
from settings.base_funs import *
import yaml



def set_up_training():
    parser = get_args()
    args = parser.parse_args()

    parser.add_argument('--model_dir', default=name(args, True), type=str, help='model directory')
    parser.add_argument('--num_cls', default=num_cls(args), type=int, help='number of classes')
    parser.add_argument('--data_dir', default=data_dir(args), type=str, help='data dir')
    parser.add_argument('--devices', default=check_cuda(args), type=list, help='check gpu devices')

    args = parser.parse_args()
    args_dict = vars(args)
    json_file = os.path.join(args.model_dir, 'args.yaml')
    with open(json_file, 'w') as f:
        yaml.dump(args_dict, f)
    return parser.parse_args()

