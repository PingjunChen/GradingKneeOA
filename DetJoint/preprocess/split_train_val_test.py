# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
import shutil, json

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)
from yolo_v2.proj_utils.local_utils import getfileinfo, mkdirs


def split_dictionary(kl_json_path):
    # Splitting category according to the R leg's kl grade
    kl_dict = {}
    with open(kl_json_path, 'r') as f:
        kl_dict = json.load(f)

    cls_dict = {
        "0": [],
        "1": [],
        "2": [],
        "3": [],
        "4": []}

    for k, v in kl_dict.items():
        if k.endswith("L"):
            continue
        cls_dict[str(int(v))].append(k[:-1])

    for k, v in cls_dict.items():
        np.random.shuffle(cls_dict[k])

    return cls_dict, kl_dict

def split_train_val_test(data_root, raw_img_dir, cls_dict, kl_dict):
    all_dict_list  = getfileinfo(raw_img_dir, ['_gt'], ['.png'], '.mat')
    cls_names = ["training", "validation", "testing"]

    for mode in cls_names:
        cur_dir = os.path.join(data_root, mode)
        mkdirs(cur_dir, erase=True)
        json_path = os.path.join(cur_dir, mode+"_kl.json")
        mode_klg_dict = {}
        mode_dict = {}
        for k, v in cls_dict.items():
            cur_list = cls_dict[k]
            if mode == "training":
                end_index = int(len(cur_list) * 0.7)
                mode_dict[k] = cls_dict[k][:end_index]
            elif mode == "validation":
                start_index = int(len(cur_list) * 0.7)
                end_index = int(len(cur_list) * 0.8)
                mode_dict[k] = cls_dict[k][start_index:end_index]
            else:
                start_index = int(len(cur_list) * 0.8)
                mode_dict[k] = cls_dict[k][start_index:]
        for k, v in mode_dict.items():
            for ele in v:
                key_left, key_right = ele + "L", ele + "R"
                mode_klg_dict[key_left] = int(kl_dict[key_left])
                mode_klg_dict[key_right] = int(kl_dict[key_right])
                shutil.copy(os.path.join(raw_img_dir, ele + ".png"), cur_dir)
                shutil.copy(os.path.join(raw_img_dir, ele + "_gt.mat"), cur_dir)

        with open(json_path, 'w') as fp:
            json.dump(mode_klg_dict, fp)

if __name__ == "__main__":
    np.random.seed(1234)
    data_root = "../data"
    raw_dir = "OAIdetreg"
    raw_img_dir = "raw_imgs"
    json_filename = "BaselineKLG.json"
    cls_dict, kl_dict = split_dictionary(os.path.join(data_root, raw_dir, json_filename))
    split_train_val_test(data_root, os.path.join(data_root, raw_dir, raw_img_dir), cls_dict, kl_dict)
