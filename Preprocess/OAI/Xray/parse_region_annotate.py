# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
import scipy.io

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.misc import imread, imsave, imresize


def parse_region_mat(mat_filepath):
    mat = scipy.io.loadmat(mat_filepath)
    contours = mat['Contours']
    boxes = []
    assert contours.shape[1] == 2, 'There should be 2 knees'
    for ind in range(contours.shape[1]):
        x_coors = contours[0, ind][0, 0:2]    # [x1, x2, x2, x1, x1]
        y_coors = contours[0, ind][1, 0:3:2]  # [y1, y1, y2, y2, y1]
        boxes.append([x_coors, y_coors])

    bb0_mid = np.mean(boxes[0][0])
    bb1_mid = np.mean(boxes[1][0])
    if bb0_mid > bb1_mid:
        boxes = boxes[::-1]
    return boxes

def crop_knee_joints(img_annotation_dir, knee_patches_dir, img_surfix='.png', annotate_surfix='_gt.mat'):
    img_list = [img for img in os.listdir(img_annotation_dir) if img.endswith(img_surfix)]
    annotate_list = [annotation for annotation in os.listdir(img_annotation_dir)
                     if annotation.endswith(annotate_surfix)]
    assert len(img_list) == len(annotate_list), "Annotation not match with image"
    expand = 0.3
    for ind, cur_img in enumerate(img_list):
        if (ind + 1) % 10 == 0:
            print("Processing {}/{} images".format(ind+1, len(img_list)))
        cur_annotate = os.path.splitext(cur_img)[0] + annotate_surfix
        bound_boxes = parse_region_mat(os.path.join(img_annotation_dir, cur_annotate))
        assert len(bound_boxes) == 2, "There are two bounding box"
        assert np.mean(bound_boxes[0][0]) < np.mean(bound_boxes[1][0]), "Box order is not correct"
        img = imread(os.path.join(img_annotation_dir, cur_img))
        assert img.shape == (2048, 2560), "The image shape is not correct"
        # Box0 as Right (expand 20%)
        box0 = bound_boxes[0]
        x0_m = np.mean(box0[0])
        x_len = box0[0][1] - box0[0][0]
        x_len_expand = x_len * (1+expand)
        x_start = int(x0_m - x_len_expand / 2.0)
        x_end = int(x0_m + x_len_expand / 2.0)
        x_start = 0 if x_start < 0 else x_start
        x_end = 2560 if x_end > 2560 else x_end
        y0_m = np.mean(box0[1])
        y_len = box0[1][1] - box0[1][0]
        y_len_expand = y_len * (1+expand)
        y_start = int(y0_m - y_len_expand / 2.0)
        y_end = int(y0_m + y_len_expand / 2.0)
        y_start = 0 if y_start < 0 else y_start
        y_end = 2048 if y_end > 2048 else y_end

        img_r = img[y_start:y_end, x_start:x_end]
        img_r = imresize(img_r, (224, 224))
        imsave(os.path.join(knee_patches_dir, os.path.splitext(cur_img)[0] + "R.png"), img_r)

        # Box1 as Left (expand 20%)
        box1 = bound_boxes[1]
        x1_m = np.mean(box1[0])
        x_len = box1[0][1] - box1[0][0]
        x_len_expand = x_len * (1+expand)
        x_start = int(x1_m - x_len_expand / 2.0)
        x_end = int(x1_m + x_len_expand / 2.0)
        x_start = 0 if x_start < 0 else x_start
        x_end = 2560 if x_end > 2560 else x_end
        y1_m = np.mean(box1[1])
        y_len = box1[1][1] - box1[1][0]
        y_len_expand = y_len * (1+expand)
        y_start = int(y1_m - y_len_expand / 2.0)
        y_end = int(y1_m + y_len_expand / 2.0)
        y_start = 0 if y_start < 0 else y_start
        y_end = 2048 if y_end > 2048 else y_end
        img_l = img[y_start:y_end, x_start:x_end]
        img_l = imresize(img_l, (224, 224))        
        imsave(os.path.join(knee_patches_dir, os.path.splitext(cur_img)[0] + "L.png"), img_l)


def batch_parse(img_dir, annotation_dir, img_surfix='.png', annotate_surfix='_gt.mat'):
    img_list = [img for img in os.listdir(img_dir) if img.endswith(img_surfix)]
    annotate_list = [annotation for annotation in os.listdir(annotation_dir)
                     if annotation.endswith(annotate_surfix)]
    assert len(img_list) == len(annotate_list), "Annotation not match with image"
    for cur_img in img_list:
        cur_annotate = os.path.splitext(cur_img)[0] + annotate_surfix
        bound_boxes = parse_region_mat(os.path.join(annotation_dir, cur_annotate))

        img = imread(os.path.join(img_dir, cur_img))
        fig, ax = plt.subplots(1)
        ax.imshow(img, cmap='gray')
        for ind in range(len(bound_boxes)):
            start_x = bound_boxes[ind][0][0]
            start_y = bound_boxes[ind][1][0]
            box_width = bound_boxes[ind][0][1] - start_x
            box_height = bound_boxes[ind][1][1] - start_y

            rect = patches.Rectangle((start_x, start_y), box_width, box_height,
                                     linewidth=3, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    # img_dir = '../../../data/DetKneeData/img_annotations'
    # annotation_dir = '../../../data/DetKneeData/img_annotations'
    # batch_parse(img_dir, annotation_dir, img_surfix='.png')

    img_annotation_dir = "../../../data/DetKneeData/img_annotations"
    knee_patches_dir = "../../../data/ClsKLData/KneePatches"
    crop_knee_joints(img_annotation_dir, knee_patches_dir)
