# -*- coding: utf-8 -*-

import os, sys, pdb
import deepdish as dd
import glob
from skimage import io
import scipy.misc as misc

# def expand_bbox()

def extract_detected_knees(data_dir, det_dir, results_dir):
    img_list = glob.glob(os.path.join(data_dir, "*.png"))
    for cur_img in img_list:
        full_name = os.path.basename(cur_img)
        cur_name = os.path.splitext(full_name)[0]
        h5_path = os.path.join(det_dir, cur_name+".h5")
        det_dict = dd.io.load(h5_path)

        img = misc.imread(cur_img)
        classes = det_dict["classes"]
        coors = det_dict["coors"]
        ind = 0
        for label, coor in zip(classes, coors):
            ind += 1
            thumb = img[coor[1]:coor[3], coor[0]:coor[2]]
            thumb = misc.imresize(thumb, (224, 224))
            save_path = os.path.join(results_dir, str(label), cur_name + '_' + str(ind) + '.png')
            misc.imsave(save_path, thumb)

if __name__ == "__main__":
    raw_img_dir = "../data/testing"
    det_result_dir = "../data/det_results"
    auto_test_dir = "../data/automatic_test"

    extract_detected_knees(raw_img_dir, det_result_dir, auto_test_dir)
