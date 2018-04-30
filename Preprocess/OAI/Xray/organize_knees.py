# -*- coding: utf-8 -*-
import os, sys, pdb
from pydaily import filesystem
import json
import shutil

def organize_kness(knee_dict, knees_dir, save_dir):
    class_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
    knee_dict = json.load(open(knee_dict))
    print("start organize")
    for key in knee_dict.keys():
        klg = str(int(knee_dict[key]))

        if klg not in class_dict.keys():
            print("unknown klg on {}".format(key))
            continue

        knee_path = os.path.join(knees_dir, key+".png")
        if os.path.exists(knee_path):
            shutil.copy2(knee_path, os.path.join(save_dir, klg))
            class_dict[klg] += 1
    print("organize finish")
    for key in class_dict.keys():
        print("In {}, there are {} cases.".format(key, class_dict[key]))


def knee_kl_splitting(det_h5_dir, knee_cat_dir, knee_split_dir):
    h5_filelist = filesystem.find_ext_files(det_h5_dir, ".h5")
    img_list = filesystem.find_ext_files(knee_cat_dir, ".png")

    for ind, img_path in enumerate(img_list):
        print("Processing {}/{}".format(ind+1, len(img_list)))
        img_id = os.path.basename(img_path)[:-5]
        img_klg = os.path.dirname(img_path)[-1]
        for h5_path in h5_filelist:
            if img_id in h5_path:
                h5_dir = os.path.dirname(h5_path)
                mode = os.path.basename(h5_dir)[:-2]
                shutil.copy2(img_path, os.path.join(knee_split_dir, mode, str(img_klg)))
                break



if __name__ == "__main__":
    # knee_dict = "../../../data/ClsKLData/baselineKLG.json"
    # knee_patches_dir = "../../../data/ClsKLData/KneePatches"
    # knee_save_dir = "../../../data/ClsKLData/KneeCategory"
    # organize_kness(knee_dict, knee_patches_dir, knee_save_dir)

    det_h5_dir = "../../../data/DetKneeData/H5"
    knee_cat_dir = "../../../data/ClsKLData/KneeCategory"
    knee_split_dir = "../../../data/ClsKLData/KneeKLsplit"
    knee_kl_splitting(det_h5_dir, knee_cat_dir, knee_split_dir)
