# -*- coding: utf-8 -*-

import os, sys, pdb
import torch
import numpy as np

def weighted_loss(outputs, labels, args):
    softmax_op = torch.nn.Softmax(1)
    prob_pred = softmax_op(outputs)

    def set_weights():
        # # weight matrix 01 (wm01)
        # init_weights = np.array([[1, 2, 3, 4, 5],
        #                          [2, 1, 2, 3, 4],
        #                          [3, 2, 1, 2, 3],
        #                          [4, 3, 2, 1, 2],
        #                          [5, 4, 3, 2, 1]], dtype=np.float)

        # weight matrix 02 (wm02)
        init_weights = np.array([[1, 3, 5, 7, 9],
                                 [3, 1, 3, 5, 7],
                                 [5, 3, 1, 3, 5],
                                 [7, 5, 3, 1, 3],
                                 [9, 7, 5, 3, 1]], dtype=np.float)

        # # weight matrix 03 (wm03)
        # init_weights = np.array([[1, 4, 7, 10, 13],
        #                          [4, 1, 4, 7, 10],
        #                          [7, 4, 1, 4, 7],
        #                          [10, 7, 4, 1, 4],
        #                          [13, 10, 7, 4, 1]], dtype=np.float)

        # # weight matrix 04 (wm04)
        # init_weights = np.array([[1, 3, 6, 7, 9],
        #                          [4, 1, 4, 5, 7],
        #                          [6, 4, 1, 3, 5],
        #                          [7, 5, 3, 1, 3],
        #                          [9, 7, 5, 3, 1]], dtype=np.float)

        adjusted_weights = init_weights + 1.0
        np.fill_diagonal(adjusted_weights, 0)

        return adjusted_weights
    cls_weights = set_weights()

    batch_num, class_num = outputs.size()
    class_hot = np.zeros([batch_num, class_num], dtype=np.float32)
    labels_np = labels.data.cpu().numpy()
    for ind in range(batch_num):
        class_hot[ind, :] = cls_weights[labels_np[ind], :]
    class_hot = torch.from_numpy(class_hot)
    class_hot = torch.autograd.Variable(class_hot).cuda()

    loss = torch.sum((prob_pred * class_hot)**2) / batch_num
    # loss = torch.mean(prob_pred * class_hot)

    return loss
