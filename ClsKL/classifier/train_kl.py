# -*- coding: utf-8 -*-

import os, sys, pdb
import  argparse
import torch
import numpy as np

from loader import data_load
from model import resnet
from train_eng import train_model
from eval_eng import eval_model

def set_args():
    parser = argparse.ArgumentParser(description='Pytorch Fine-Tune Resnet Training')
    parser.add_argument('--seed',                  type=int, default=1234)
    parser.add_argument('--net_type',              type=str, default='resnet')
    parser.add_argument('--depth',                 type=int, default=34)
    parser.add_argument('--lr',                    type=float, default=5.0e-4)
    parser.add_argument('--lr_decay_epoch',        type=int, default=5)
    parser.add_argument('--num_epoch',             type=int, default=12)
    parser.add_argument('--batch_size',            type=int, default=32)
    parser.add_argument('--weight_decay',          type=float, default=5.0e-4)
    parser.add_argument('--data_dir',              type=str, default='../../data/ClsKLData/mannual_crop/')
    parser.add_argument('--model_dir',             type=str, default='../../data/ClsKLData/models/')
    parser.add_argument('--pretrained',            type=bool, default=True)
    parser.add_argument('--cuda_id',               type=int, default=3)
    parser.add_argument('--optim',                 type=str, default='SGD')
    parser.add_argument('--wloss',                 type=bool, default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print('--Phase 0: Argument settings--')
    args = set_args()
    np.random.seed(args.seed)
    args.best_model_name = 'resnet{}-{}-'.format(args.depth, args.optim)
    print("**seed is {}".format(args.seed))

    print('--Phase 1: Data prepration--')
    dset_loaders, dset_size, num_class = data_load(args)
    args.num_class = num_class

    print('--Phase 2: Model setup--')
    model = resnet(args)
    if torch.cuda.is_available():
        model.cuda(args.cuda_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    print('--Phase 3: Model training--')
    train_model(args, model, dset_loaders, dset_size)

    # print('--Phase 4: Evaluate model--')
    # phase='val'
    # eval_model(args, phase, dset_loaders, dset_size)