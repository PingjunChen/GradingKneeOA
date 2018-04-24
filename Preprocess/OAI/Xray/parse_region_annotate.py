import os, sys, pdb
import scipy.io

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.misc import imread


def parse_region_mat(mat_filepath):
    mat = scipy.io.loadmat(mat_filepath)
    contours = mat['Contours']
    boxes = []
    assert contours.shape[1] == 2, 'There should be 2 knees'
    for ind in range(contours.shape[1]):
        x_coors = contours[0, ind][0, 0:2]    # [x1, x2, x2, x1, x1]
        y_coors = contours[0, ind][1, 0:3:2]  # [y1, y1, y2, y2, y1]
        boxes.append([x_coors, y_coors])

    return boxes

def batch_parse(img_dir, annotation_dir, img_surfix='.png', annotate_surfix='_gt.mat'):
    img_list = [img for img in os.listdir(img_dir) if img.endswith(img_surfix)]
    annotate_list = [annotation for annotation in os.listdir(annotation_dir)
                     if annotation.endswith(annotate_surfix)]
    assert len(img_list) == len(annotate_list), "Annotation not match with image"
    for cur_img in img_list:


        pdb.set_trace()
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


if __name__ == '__main__':
    img_dir = '/media/pingjun/PingjunOAI/OAIdata/KneeJointsAnnotation/KneeAggregation'
    annotation_dir = '/media/pingjun/PingjunOAI/OAIdata/KneeJointsAnnotation/KneeAggregation'

    batch_parse(img_dir, annotation_dir, img_surfix='.png')
