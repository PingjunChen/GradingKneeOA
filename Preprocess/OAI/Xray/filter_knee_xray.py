# -*- coding: utf-8 -*-

import os, sys, pdb
import pandas as pd
import shutil

def extract_knee_xray(root_dir, excel_file, dst_dir):
    excel_path = os.path.join(root_dir, excel_file)
    df24m = pd.read_csv(excel_path)
    df_knee = df24m[df24m["SeriesDescription"] == "Bilateral PA Fixed Flexion Knee "]
    # pdb.set_trace()

    xray_dirs = df_knee.Folder.values.tolist()
    xray_ids = df_knee.ParticipantID.values.tolist()

    num_xray = len(xray_dirs)
    assert len(xray_dirs) == len(xray_ids), "The number of dir and id are not matching"
    print("There are {} knees in total".format(num_xray))
    count = 0
    for xray_dir, pid in zip(xray_dirs, xray_ids):
        xray_path = os.path.join(root_dir, xray_dir, "001")
        shutil.copy(xray_path, os.path.join(dst_dir, str(pid)+".dcm"))
        if os.path.exists(xray_path):
            count += 1
        if count % 50 == 0:
            print("Copy {}/{}".format(count, num_xray))

    print("There are {} files in total".format(count))


if __name__ == "__main__":
    root_dir = "/media/pingjun/lab_drive_SG/OAI/Xrays/96m"
    excel_file = "contents.csv"
    dst_dir = "/media/pingjun/PingjunLab/KneeData/OAI/Xray/96m"

    extract_knee_xray(root_dir, excel_file, dst_dir)
