# -*- coding: utf-8 -*-
import os, sys, pdb
from pydaily import filesystem
import json
import shutil

def organize_kness(knee_dict, Knees_dir, save_dir):
    class_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
    knee_dict = json.load(open(knee_dict))
    print("start organize")
    for key in knee_dict.keys():
        klg = str(int(knee_dict[key]))

        if klg not in class_dict.keys():
            print("unknown klg on {}".format(key))
            continue

        knee_path = os.path.join(Knees_dir, key+".png")
        if os.path.exists(knee_path):
            shutil.copy2(knee_path, os.path.join(save_dir, klg))
            class_dict[klg] += 1
    print("organize finish")
    for key in class_dict.keys():
        print("In {}, there are {} cases.".format(key, class_dict[key]))


if __name__ == "__main__":
    knee_dict = r"../data/BaselineKLG.json"
    Knees_dir = r"C:\KneeXray\KneePatches"
    save_dir = r"../data/Knees"

    organize_kness(knee_dict, Knees_dir, save_dir)
