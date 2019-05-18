# -*- coding: utf-8 -*-

import os, sys
import numpy as np

def build_cof_mat(num, poly_num=2):
    assert num > 2, "num should bigger than 2"
    cof_mat = np.zeros((num, num), dtype=np.float32)
    for i in range(0, num):
        for j in range(0, num):
            cof_mat[i, j] = np.power(np.absolute(i-j), poly_num)

    return cof_mat


def ordinal_mse(confusion_matrix, poly_num=2):
    category_num = confusion_matrix.shape[0]
    cof_mat = build_cof_mat(category_num, poly_num=poly_num)

    err_mat = np.multiply(confusion_matrix, cof_mat)
    mse_val = np.sum(err_mat) * 1.0 / np.sum(confusion_matrix)

    return mse_val
