# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
from sklearn.cluster import KMeans

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)
from yolo_v2.datasets.knee import Knee


if __name__ == '__main__':
    data_root = "../data/"
    train_set = Knee(data_root, "training")
    all_bbox = train_set.get_all_bbox()

    col_size_list = all_bbox[:,2] - all_bbox[:,0]
    row_size_list = all_bbox[:,3] - all_bbox[:,1]
    dim_array = np.stack([col_size_list,row_size_list], axis=1)

    # Clustered centered image dims
    kmeans = KMeans(n_clusters=3, random_state=0).fit(dim_array)
    centers = kmeans.cluster_centers_
    print("Clustered centers are:")
    print(centers)
