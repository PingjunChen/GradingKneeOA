# -*- coding: utf-8 -*-

import os, sys, pdb
import matplotlib.pyplot as plt
import numpy as np

def ablation_compare():
    num_box = [1, 2, 3, 4, 5, 6]
    with_wd = [0.922, 0.868, 0.902, 0.872, 0.871, 0.855]
    no_wd = [0.911, 0.864, 0.865, 0.856, 0.851, 0.828]

    plt.plot(num_box, with_wd, marker='D', linestyle='--', color='r', linewidth=3.0,
             label='use weight decay')
    plt.plot(num_box, no_wd, marker='o', linestyle='--', color='b', linewidth=3.0,
             label='no weight decay')
    plt.xlim(0.6, 6.4)
    plt.ylim(0.8, 0.96)
    plt.legend()
    # plt.tight_layout()

    plt.xlabel('Number of bounding box')
    plt.ylabel('Recall')
    # plt.title('Knee joint detection')
    plt.grid(True)
    plt.savefig("yolo_para.pdf")
    plt.show()

if __name__ == "__main__":
    ablation_compare()
