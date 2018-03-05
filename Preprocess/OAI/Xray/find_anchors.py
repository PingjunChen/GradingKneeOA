# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
from scipy import io
from sklearn.cluster import KMeans
from pydaily import filesystem

def collect_boxes(xrays):
    mat_list = filesystem.find_ext_files(xrays, ".mat")

    # max_ratio, min_ratio = 1.12, 0.88
    # max_count, min_count = 0, 0

    boxes = []
    for cur_mat in mat_list:
        mat = io.loadmat(cur_mat)
        contours = mat['Contours']
        assert contours.shape[1] == 2, "{} no two boxes".format(cur_mat)
        # Extract bounding box
        for ind in range(contours.shape[1]):
            x_coors = contours[0, ind][0, 0:2]    # [x1, x2, x2, x1, x1]
            y_coors = contours[0, ind][1, 0:3:2]  # [y1, y1, y2, y2, y1]

            box_h = y_coors[1] - y_coors[0]
            box_w = x_coors[1] - x_coors[0]
            boxes.append([box_h, box_w])

            # if (box_h / box_w) > max_ratio:
            #     max_count += 1
            # if (box_h / box_w) < min_ratio:
            #     min_count += 1

    # print("Max count num is: {}".format(max_count))
    # print("Min count num is: {}".format(min_count))

    boxes = np.asarray(boxes, np.float32)
    return boxes

if __name__ == '__main__':
    pingjun_xrays = r"C:\KneeXray\KneeAggregation"
    boxes = collect_boxes(pingjun_xrays)

    # Clustered centered image dims
    kmeans = KMeans(n_clusters=6, random_state=0).fit(boxes)
    centers = kmeans.cluster_centers_
    print("Clustered centers are:")
    # print(centers)
    for cur_center in centers:
        ratio = cur_center[0] / cur_center[1]
        print("H: {:.2f}, W: {:.2f}, ratio: {:.3f}".format(cur_center[0], cur_center[1], ratio))

    # #Ratio
    # [1.12, 1.08, 1.04, 1.0, 0.96, 0.92]
    # Max ratio is: 1.2653061224489797
    # Min ratio is: 0.7877813504823151
