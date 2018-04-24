# -*- coding: utf-8 -*-

import os, sys
import numpy as np

class config:
    def __init__(self):
        label_names = ["0", "1", "2", "3", "4"]
        num_classes = len(label_names)

        anchors = np.asarray([[ 65.3,  67.7],
                              [ 62.4,  57.5],
                              [ 75.0,  72.3]])
        num_anchors = len(anchors)

        # IOU scale
        iou_thresh = 0.6
        object_scale = 5.

        # iou_thresh = 0.75
        # object_scale = 3.
        noobject_scale = 1.

        # BOX scale
        coord_scale = 1.

        # CLS scale
        # class_scale = 1.0  # for kl classification
        class_scale = 0.0    # only for detection

        w_loss = True        # for weighted loss
        self.__dict__.update(locals())

cfg = config()
