# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)
from yolo_v2.datasets.knee import Knee

if __name__ == '__main__':
    data_root = "../../data/DetKneeData"
    train_set = Knee(data_root, "train")

    pixel_mean = train_set.get_mean_pixel()
    pixel_var = train_set.get_var_pixel()
    print("Mean gray value is {}, variance is {}".format(pixel_mean, pixel_var))
