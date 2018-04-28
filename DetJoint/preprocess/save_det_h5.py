# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
import h5py, json
from scipy.io import loadmat
import deepdish as dd
import scipy.misc as misc
import numpy as np

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)
from yolo_v2.proj_utils.local_utils import getfileinfo, mkdirs
from yolo_v2.proj_utils.local_utils import writeImg, imread, imresize


# Load annotated bounding box mat file
def load_mat(thismatfile, contourname_list=['Contours']):
    # First try load using h5py; then try using scipy.io.loadmat
    try:
        mat_file = h5py.File(thismatfile, 'r')
        for contourname in contourname_list:
            if contourname in list(mat_file.keys()):
                contour_mat = [np.transpose(mat_file[element[0]][:])
                               for element in mat_file[contourname]]
                break
        mat_file.close()
    except:
        loaded_mt = loadmat(thismatfile)
        for contourname in contourname_list:
            if contourname in loaded_mt.keys():
                contour_mat = []
                cnts = loaded_mt[contourname].tolist()
                if len(cnts) > 0:
                    contour_mat = cnts[0]
                break

    return contour_mat


def get_bbox(contour_mat):
    numCell = len(contour_mat)
    bbox_list = []
    for icontour in range(0, numCell):
        thiscontour = contour_mat[icontour]
        xcontour = np.reshape(thiscontour[0,:], (1,-1) )
        ycontour = np.reshape(thiscontour[1,:], (1,-1) )

        x_min, x_max = np.min(xcontour), np.max(xcontour)
        y_min, y_max = np.min(ycontour), np.max(ycontour)
        bbox_list.append([x_min, y_min, x_max, y_max])
    return bbox_list


# Resize bounding box to a certain ratio
def resize_mat(contour_mat, resize_ratio):
    numCell = len(contour_mat)
    res_contour = []
    for icontour in range(0, numCell):
        thiscontour = contour_mat[icontour]
        res_contour.append([ele*resize_ratio for ele in thiscontour] )
    return res_contour


def save_h5(data_root, h5_dir, mode, ratio=0.125):
    all_dict_list  = getfileinfo(os.path.join(data_root, mode), ['_gt'], ['.png'], '.mat')
    mode_dir = os.path.join(data_root, h5_dir, mode+"H5")
    mkdirs(mode_dir, erase=True)

    klg_dict = {}
    with open(os.path.join(data_root, mode, mode + "_kl.json"), 'r') as f:
        kl_dict = json.load(f)

    for ele in all_dict_list:
        info_dict = {}

        img_path = ele["thisfile"]
        cur_img = imread(img_path)
        mat_path = ele["thismatfile"]
        contour_mat = load_mat(mat_path)
        cur_bbox =  get_bbox(contour_mat)
        assert len(cur_bbox) == 2, "Error, there are not 2 bbox"
        pat_id= os.path.splitext(os.path.basename(img_path))[0]
        key_r, key_l = pat_id + "R", pat_id + "L"


        resized_img = imresize(cur_img, ratio)
        assert resized_img.shape[:2] == (256, 320)
        if cur_bbox[0][0] > cur_bbox[1][0]:
            cur_bbox = cur_bbox[::-1]
        resized_bbox = resize_mat(cur_bbox, ratio)
        classes = [kl_dict[key_r], kl_dict[key_l]]

        info_dict["images"] = resized_img
        info_dict["gt_boxes"] = resized_bbox
        info_dict["gt_classes"] = classes
        info_dict["dontcare"] = [0] * len(classes)
        info_dict["origin_im"] = pat_id

        dd.io.save(os.path.join(mode_dir, pat_id+".h5"), info_dict)



if __name__ == "__main__":
    np.random.seed(1234)

    knee_data_root = "../../data"
    data_root = os.path.join(knee_data_root, "DetKnee")
    h5_dir = "H5"

    modes = ["train", "val", "test"]
    for mode in modes:
        print("Save {} dataset".format(mode))
        save_h5(data_root, h5_dir, mode)
