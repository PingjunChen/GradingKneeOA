# -*- coding: utf-8 -*-

import os, sys, pdb
from scipy import io
import shutil
from pydaily import filesystem


def check_boxes(mat_filepath):
    mat = io.loadmat(mat_filepath)
    contours = mat['Contours']
    if contours.shape[1] != 2:
        return False

    # Extract bounding box
    boxes = []
    for ind in range(contours.shape[1]):
        x_coors = contours[0, ind][0, 0:2]    # [x1, x2, x2, x1, x1]
        y_coors = contours[0, ind][1, 0:3:2]  # [y1, y1, y2, y2, y1]

        box_w = x_coors[1] - x_coors[0]
        box_h = y_coors[1] - y_coors[0]
        if box_w < 320 or box_w > 720 or box_h < 320 or box_h > 720:
            print("box_w {}, box_h {}".format(box_w, box_h))
            return False
        boxes.append([x_coors, y_coors])

    if len(boxes) != 2:
        return False

    return True

def check_all_annotations(xrays):
    filelist = filesystem.find_ext_files(xrays, ".png")
    matlist = filesystem.find_ext_files(xrays, ".mat")
    assert len(filelist) == len(matlist), "file and mat not match..."
    print("There are {} files".format(len(filelist)))

    for cur_file in filelist:
        file_dir = os.path.dirname(cur_file)
        file_name = os.path.basename(cur_file)
        mat_name = os.path.splitext(file_name)[0] + "_gt.mat"
        mat_path = os.path.join(file_dir, mat_name)
        if mat_path not in matlist:
            print("{} no corresponding mat file...".format(file_name))
            continue

        box_flag = check_boxes(mat_path)
        if box_flag == False:
            print("{} annotation error...".format(file_name))
            shutil.move(mat_path, r"C:\KneeXray\ToRevise")
            shutil.move(cur_file, r"C:\KneeXray\ToRevise")


if __name__ == '__main__':
    pingjun_xrays = r"C:\KneeXray\KneeAggregation"
    check_all_annotations(pingjun_xrays)
