# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
from sklearn.cluster import KMeans

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)
from yolo_v2.datasets.knee import Knee


if __name__ == '__main__':
    data_root = "../../data/DetKneeData"
    train_set = Knee(data_root, "train")
    all_bbox = train_set.get_all_bbox()

    # [x1, y1, x2, y2]
    col_size_list = all_bbox[:,2] - all_bbox[:,0]
    row_size_list = all_bbox[:,3] - all_bbox[:,1]
    dim_array = np.stack([col_size_list, row_size_list], axis=1)

    # Clustered centered image dims
    cluster_num = 6
    # cluster_num = 5

    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(dim_array)
    centers = kmeans.cluster_centers_
    print("{} clustered centers:".format(cluster_num))
    for center in centers:
        print("W: {:.2f}, H: {:.2f}, ratio: {:.2f}".format(
            center[0], center[1], center[1]/center[0]))
