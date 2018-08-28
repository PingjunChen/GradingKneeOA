# -*- coding: utf-8 -*-

import os, sys, pdb
import argparse
import torch
from torch.utils import data
import torchvision.transforms as standard_transforms
import numpy as np

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)

from yolo_v2.proj_utils.local_utils import mkdirs
from yolo_v2.cfgs.config_knee import cfg
from yolo_v2.darknet import Darknet19
from yolo_v2.datasets.knee import Knee
from yolo_v2.train_yolo import train_eng


def set_args():
    # Arguments settinge
    parser = argparse.ArgumentParser(description="Knee Bone Detection")
    parser.add_argument('--batch_size',      type=int,   default=8,            help='batch size.')
    parser.add_argument('--maxepoch',        type=int,   default=500,          help='number of epochs to train')
    parser.add_argument('--lr',              type=float, default=2.0e-4,       help='learning rate')
    parser.add_argument('--lr_decay',        type=float, default=0.8,          help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=list,  default=[60, 120, 180, 240, 300, 360, 420, 480],
                        help='decay the learning rate at this epoch')
    parser.add_argument('--weight_decay',    type=float, default=0.0,          help='weight decay for training')
    parser.add_argument('--momentum',        type=float, default=0.9,          help='SGD momentum (default: 0.9)')
    parser.add_argument('--display_freq',    type=int,   default=10,           help='plot the results per batches')
    parser.add_argument('--save_freq',       type=int,   default=10,           help='how frequent to save the model')
    parser.add_argument('--device-id',       type=int,   default=0)
    parser.add_argument('--model-name',      type=str,   default='kneedet')
    parser.add_argument('--seed',            type=int,   default=1234)

    args = parser.parse_args()
    return args


if  __name__ == '__main__':
    args = set_args()
    np.random.seed(args.seed)

    # Data and Model settings
    data_root = "../../data/DetKneeData"
    model_root = os.path.join(data_root, args.model_name)
    mkdirs(model_root, erase=True)

    # Replace as mean and std
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(cfg.rgb_mean, cfg.rgb_var)])
    train_dataset = Knee(data_root, "train", transform=input_transform)
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataset = Knee(data_root, "val", transform=input_transform)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size)

    # Set Darknet
    net = Darknet19(cfg)

    # CUDA Settings
    cuda_avail = torch.cuda.is_available()
    print("\n==== Starting training ====\n" + "===="*20)
    if cuda_avail:
        print("CUDA {} in use".format(args.device_id))
        net.cuda(args.device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    else:
        print("CPU in use")

    # print ('>> START training ')
    train_eng(train_dataloader, val_dataloader, model_root, net, args)
