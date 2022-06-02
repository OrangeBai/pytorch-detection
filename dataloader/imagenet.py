from config import *
import os
import torch
import torch.utils.data as data
from torchvision.transforms import *
import torchvision.datasets as datasets


def get_loaders(args):
    data_dir = os.path.join(DATA_PATH, 'ImageNet')
    train_dir = os.path.join(data_dir, 'train')

    train_transform = Compose([Resize(args.DATA.img_size), RandomResizedCrop(args.DATA.crop_size),
                               RandomHorizontalFlip(), ToTensor()])
    val_transform = Compose([Resize(args.DATA.img_size), CenterCrop(args.DATA.crop_size), transforms.ToTensor()])
    train_dataset = datasets.MNIST(train_dir, train=True, transform=train_transform, download=True)
    test_dataset = datasets.MNIST(DATA_PATH, train=False, transform=val_transform, download=True)

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
