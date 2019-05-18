# -*- coding: utf-8 -*-

import os, sys, pdb
import  argparse
import torch
import numpy as np

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)

from utils.loader import data_load
# from utils.model import resnet
from utils.model import cls_model
from utils.train_eng import train_model
from utils.eval_eng import eval_model


def set_args():
    parser = argparse.ArgumentParser(description='Pytorch Fine-Tune Resnet Training')
    parser.add_argument('--net_type',              type=str, default="vgg")
    parser.add_argument('--depth',                 type=str, default="19")
    parser.add_argument('--lr',                    type=float, default=5.0e-4)
    parser.add_argument('--lr_decay_epoch',        type=int, default=5)
    parser.add_argument('--num_epoch',             type=int, default=12)
    parser.add_argument('--batch_size',            type=int, default=32)
    parser.add_argument('--weight_decay',          type=float, default=5.0e-4)
    parser.add_argument('--data_dir',              type=str, default='../../data/ClsKLData/kneeKL224')
    parser.add_argument('--model_dir',             type=str, default='../../data/ClsKLData/models/cmpLoss')
    parser.add_argument('--pretrained',            type=bool, default=True)
    parser.add_argument('--cuda_id',               type=int, default=0)
    parser.add_argument('--optim',                 type=str, default='SGD')
    parser.add_argument('--wloss',                 type=int, default=1)
    parser.add_argument('--session',               type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('--Phase 0: Argument settings--')
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    args.best_model_name = '{}-{}-{}-{}'.format(args.net_type, args.depth, args.optim, args.wloss)
    # import pdb; pdb.set_trace()

    print('--Phase 1: Data prepration--')
    dset_loaders, dset_size, num_class = data_load(args)
    args.num_class = num_class

    print('--Phase 2: Model setup--')
    model = cls_model(args)
    model.cuda()

    print('--Phase 3: Model training--')
    train_model(args, model, dset_loaders, dset_size)

    # print('--Phase 4: Evaluate model--')
    # phase='val'
    # eval_model(args, phase, dset_loaders, dset_size)
