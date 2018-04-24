# -*- coding: utf-8 -*-

import os, sys, pdb
import  argparse
import torch
from torchvision import models
import numpy as np

from loader import data_load
from eval_eng import eval_model, gen_vis_loc


def set_args():
    parser = argparse.ArgumentParser(description='Pytorch Fine-Tune Resnet Testing')
    parser.add_argument('--batch_size',            type=int, default=32)
    parser.add_argument('--data_dir',              type=str, default='../data/')
    parser.add_argument('--save_dir',              type=str, default='../vis/data/heatmap')
    parser.add_argument('--model_dir',             type=str, default='../data/cls_models/chen_model')
    parser.add_argument('--best_model_name',       type=str, default='resnet34-0.635-0.568.pth')

    # parser.add_argument('--model_dir',             type=str, default='../data/cls_models/linlin_model')
    # parser.add_argument('--best_model_name',       type=str, default='resnet34-SGD-mstdTrue-wlossTrue-12-0.597-0.64-0.731.pth')

    parser.add_argument('--cuda_id',               type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    args.best_model_path = os.path.join(args.model_dir, args.best_model_name)

    dset_loaders, dset_size, num_class = data_load(args)
    # phase="automatic_test"
    phase="test"

    # # Evaluate model
    # print('---Evaluate model : {}--'.format(phase))
    # eval_model(args, phase, dset_loaders, dset_size)

    # Generate saliency visulization
    gen_vis_loc(args, phase, dset_loaders, dset_size, args.save_dir)
