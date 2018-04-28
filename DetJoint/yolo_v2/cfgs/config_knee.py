# -*- coding: utf-8 -*-

import os, sys
import numpy as np

class config:
    def __init__(self):
        self.label_names = ["0", "1", "2", "3", "4"]
        self.num_classes = len(self.label_names)

        # w * h
        self.anchors = np.asarray([[ 65.3,  67.7],
                                   [ 62.4,  57.5],
                                   [ 75.0,  72.3]])
        self.num_anchors = len(self.anchors)

        # Image mean and std
        self.rgb_mean = [0.5, 0.5, 0.5]
        self.rgb_var = [0.5, 0.5, 0.5]

        # IOU scale
        self.iou_thresh = 0.6
        self.object_scale = 5.
        self.noobject_scale = 1.

        # BOX scale
        self.coord_scale = 1.

        # CLS scale
        # self.class_scale = 1.0  # for kl classification
        self.class_scale = 0.0    # only for detection

        # Regulable Ordinal Loss
        self.w_loss = False        # for weighted loss

        # Test JI index
        self.JIthresh = 0.75

cfg = config()
