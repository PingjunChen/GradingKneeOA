# -*- coding: utf-8 -*-

import os, sys, pdb
import torch
from torchvision import models


def resnet(args):
    if args.depth == 18:
        model = models.resnet18(pretrained = args.pretrained)
    elif args.depth == 34:
        model = models.resnet34(pretrained = args.pretrained)
    elif args.depth == 50:
        model = models.resnet50(pretrained = args.pretrained)
    elif args.depth == 101:
        model = models.resnet101(pretrained = args.pretrained)
    else:
        model = models.resnet152(pretrained = args.pretrained)

    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, args.num_class)

    return model
