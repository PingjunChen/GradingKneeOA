# -*- coding: utf-8 -*-

import os, sys, pdb
import deepdish as dd
import glob
from skimage import io
import scipy.misc as misc

# def expand_bbox()

def extract_detected_knees(data_dir, det_dir, results_dir, expand=0.3):
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
            # thumb = img[coor[1]:coor[3], coor[0]:coor[2]]

            x_len = coor[2] - coor[0]
            x_len_expand = x_len * (1 + expand)
            x_mid = (coor[2] + coor[0]) / 2
            x_start = int(x_mid - x_len_expand / 2.0)
            x_end = int(x_mid + x_len_expand / 2.0)
            x_start = 0 if x_start < 0 else x_start
            x_end = 2560 if x_end > 2560 else x_end

            y_len = coor[3] - coor[1]
            y_len_expand = y_len * (1 + expand)
            y_mid = (coor[3] + coor[1]) / 2
            y_start = int(y_mid - y_len_expand / 2.0)
            y_end = int(y_mid + y_len_expand / 2.0)
            y_start = 0 if y_start < 0 else y_start
            y_end = 2048 if y_end > 2048 else y_end

            thumb = img[y_start:y_end, x_start:x_end]
            thumb = misc.imresize(thumb, (299, 299))
            save_path = os.path.join(results_dir, str(label), cur_name + '_' + str(ind) + '.png')
            misc.imsave(save_path, thumb)

if __name__ == "__main__":
    raw_img_dir = "../../data/DetKneeData/test"
    det_result_dir = "../../data/DetKneeData/det_results"
    auto_test_dir = "../../data/DetKneeData/automatic_test299"

    extract_detected_knees(raw_img_dir, det_result_dir, auto_test_dir)
