from torchvision import transforms, datasets
import torch.utils.data as data
from config import *
import numpy as np


def get_loaders(args):
    mean, std = (0.1307,), (0.3081,)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = datasets.MNIST(DATA_PATH, train=True, transform=data_transform, download=True)
    test_dataset = datasets.MNIST(DATA_PATH, train=False, transform=data_transform, download=True)

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, test_loader

# def get_test_set(*args):
#     """
#     Load test dataset according to labels:
#     'all': all data
#     1 : data with label 1
#     2 : data with label 2 ......
#     :param args: labels:
#                     'all': all data
#                     1 : data with label 1
#                     2 : data with label 2 ......
#     :return: a collection of data loaders
#     """
#     test_transform = transforms.Compose([transforms.ToTensor()])
#     test_loaders = []
#     test_sets = []
#     for arg in args:
#         if arg == 'all':
#             test_dataset = datasets.MNIST(DATA_PATH, train=False, transform=test_transform, download=True)
#         elif arg in range(9):
#             test_dataset = datasets.MNIST(DATA_PATH, train=False, transform=test_transform, download=True)
#             test_dataset.data = test_dataset.data[np.where(test_dataset.targets.numpy() == arg)][:1000]
#             test_dataset.targets = test_dataset.targets[np.where(test_dataset.targets.numpy() == arg)][:1000]
#         else:
#             raise ValueError("Cannot find dataset %s" % str(args))
#
#         test_sets += [test_dataset]
#         test_loaders += [torch.utils.data.DataLoader(
#             dataset=test_dataset,
#             batch_size=128,
#             shuffle=False,
#             pin_memory=True,
#             num_workers=2,
#         )]
#     return test_loaders, test_sets
