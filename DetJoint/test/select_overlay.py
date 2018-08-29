# -*- coding: utf-8 -*-

import os, sys, pdb
import deepdish as dd
import json

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)

from yolo_v2.knee_utils import overlay_bbox_iou
from yolo_v2.proj_utils.local_utils import writeImg

if  __name__ == '__main__':
    # selection = ["9559547", "9970801", "9023935", "9488966", "9994363", "9103811", "9313330", "9607454"]
    select_json_path = "../../data/DetKneeData/results/selection.json"
    h5dir = "../../data/DetKneeData/H5/testH5"

    # load json file
    json_dict = json.load(open(select_json_path))
    for ele in json_dict:
        h5_file = os.path.join(h5dir, ele+".h5")
        cur_item = dd.io.load(h5_file)
        cur_img = cur_item["images"]
        cur_det_boxes = json_dict[ele]["det"]
        cur_gt_boxes = json_dict[ele]["gt"]

        overlaid_img = overlay_bbox_iou(cur_img, cur_det_boxes, cur_gt_boxes)

        writeImg(overlaid_img, os.path.join(os.path.dirname(select_json_path), ele+'.png'))
