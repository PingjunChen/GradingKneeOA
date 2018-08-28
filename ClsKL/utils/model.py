# -*- coding: utf-8 -*-

import os, sys, pdb
import torch
import torch.nn as nn
from torchvision import models


def cls_model(args):
    if args.net_type == "resnet":
        if args.depth == "18":
            model = models.resnet18(pretrained = args.pretrained)
        elif args.depth == "34":
            model = models.resnet34(pretrained = args.pretrained)
        elif args.depth == "50":
            model = models.resnet50(pretrained = args.pretrained)
        elif args.depth == "101":
            model = models.resnet101(pretrained = args.pretrained)
        elif args.depth == "152":
            model = models.resnet152(pretrained = args.pretrained)
        else:
            return None

        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, args.num_class)

    elif args.net_type == "vgg":
        if args.depth == "16":
            model = models.vgg16(pretrained = args.pretrained)
        elif args.depth == "19":
            model = models.vgg19(pretrained = args.pretrained)
        elif args.depth == "16bn":
            model = models.vgg16_bn(pretrained = args.pretrained)
        elif args.depth == "19bn":
            model = models.vgg19_bn(pretrained = args.pretrained)
        else:
            return None

        num_ftrs = model.classifier[6].in_features
        feature_model = list(model.classifier.children())
        feature_model.pop()
        feature_model.append(nn.Linear(num_ftrs, args.num_class))
        model.classifier = nn.Sequential(*feature_model)

    elif args.net_type == "densenet":
        if args.depth == "121":
            model = models.densenet121(pretrained = args.pretrained)
        elif args.depth == "169":
            model = models.densenet169(pretrained = args.pretrained)
        elif args.depth == "201":
            model = models.densenet201(pretrained = args.pretrained)
        else:
            return None

        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, args.num_class)

    elif args.net_type == "inception":
        if args.depth == "v3":
            model = models.inception_v3(pretrained = args.pretrained)
        else:
            return None

        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, args.num_class)

    else:
        return None

    return model
