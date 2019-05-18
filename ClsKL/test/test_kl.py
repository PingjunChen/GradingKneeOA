# -*- coding: utf-8 -*-

import os, sys, pdb
import  argparse
import torch
from torchvision import models
import numpy as np

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)

from utils.loader import data_load
from utils.eval_eng import eval_model, gen_vis_loc, gen_grad_cam


def set_args():
    parser = argparse.ArgumentParser(description='Pytorch Fine-Tune KL grading testing')
    parser.add_argument('--cuda-id',               type=int, default=0)
    parser.add_argument('--batch-size',            type=int, default=16)
    parser.add_argument('--data-dir',              type=str, default='../../data/ClsKLData/kneeKL224')
    parser.add_argument('--model_dir',             type=str, default='')
    parser.add_argument('--best_model_name',       type=str, default='')
    # parser.add_argument('--save_dir',              type=str, default='../../data/ClsKLData/models/CAMs/')
    parser.add_argument('--phase',                 type=str, default="test", choices=["test", "auto_test"])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    args.best_model_path = os.path.join(args.model_dir, args.best_model_name)
    assert os.path.exists(args.best_model_path), "Model doesnot exist"

    dset_loaders, dset_size, num_class = data_load(args)

    # Evaluate model
    print('---Evaluate model : {}--'.format(args.phase))
    eval_model(args, args.phase, dset_loaders, dset_size)

    # Generate saliency visulization
    # gen_vis_loc(args, phase, dset_loaders, dset_size, args.save_dir)
    # gen_grad_cam(args, phase, dset_loaders, dset_size, args.save_dir)
