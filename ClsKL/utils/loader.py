# -*- coding: utf-8 -*-

import os, sys, pdb
import torch
from torchvision import transforms

from .knee_sets import ImageFolder


def data_load(args):
    pixel_mean, pixel_std = 0.66133188,  0.21229856
    phases = ['train', 'val', 'test', 'auto_test']
    # phases = ['train', 'val', 'test', 'auto_test']
    data_transform = {
        'train': transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'auto_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'most_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'most_auto_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ])
    }

    dsets = {x: ImageFolder(os.path.join(args.data_dir, x), data_transform[x]) for x in phases}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,
            shuffle=(x=='train'), num_workers=4) for x in phases}
    dset_classes = dsets['train'].classes
    dset_size = {x: len(dsets[x]) for x in phases}
    num_class = len(dset_classes)

    return dset_loaders, dset_size, num_class
