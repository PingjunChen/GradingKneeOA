# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
import cv2

from .proj_utils.torch_utils import to_device
from .utils.cython_yolo import yolo_to_bbox
from .proj_utils.local_utils import change_val

# knee detection and classification forward
def knee_det_cls(cls, img, cfg=None):
    results = {'bbox':[],'iou':[], 'prob':[]}

    batch_data = to_device(img, cls.device_id, volatile=True)
    bbox_pred, iou_pred, prob_pred  = cls.forward(batch_data)

    bbox_pred = bbox_pred.cpu().data.numpy()
    iou_pred = iou_pred.cpu().data.numpy()
    prob_pred = prob_pred.cpu().data.numpy()

    H, W = cls.out_size
    x_ratio, y_ratio = cls.x_ratio, cls.y_ratio

    bbox_pred = yolo_to_bbox(
                np.ascontiguousarray(bbox_pred, dtype=np.float),
                np.ascontiguousarray(cfg.anchors, dtype=np.float),
                H, W, x_ratio, y_ratio)

    results['bbox'] = np.array(bbox_pred)
    results['iou'] = np.array(iou_pred)
    results['prob'] = np.array(prob_pred)

    return results


# Evaluate knee detection and classification
def evaluate_det_cls(gt_boxes, gt_classes, pr_boxes, pr_classes, num=5, overlap_ratio=0.75):
    t_box_num = 0
    all_box_num = len(pr_boxes)
    pred_matrix = np.zeros((num, num), dtype=np.int)
    overlap_list = []

    for gt_box, gt_class in zip(gt_boxes, gt_classes):
        for pr_box, pr_class in zip(pr_boxes, pr_classes):
            cur_overlap = evaluate_box_JI(gt_box, pr_box)
            # cur_overlap = evaluate_box_dice(gt_box, pr_box)
            if cur_overlap >= overlap_ratio:
                pred_matrix[gt_class, pr_class] += 1
                t_box_num += 1
                overlap_list.append(cur_overlap)

    return pred_matrix, t_box_num, all_box_num, overlap_list


def overlay_bbox_iou(img, pred_boxes, gt_boxes, len=3):
    # Draw gt boxes
    for bb in gt_boxes:
        x_min_, y_min_, x_max_, y_max_ = bb
        x_min_, y_min_, x_max_, y_max_ = int(x_min_),int( y_min_), int(x_max_), int(y_max_)
        img[:,:,0] = change_val(img[:,:,0], 255, len, x_min_, y_min_, x_max_, y_max_)
        img[:,:,1] = change_val(img[:,:,1], 0, len,  x_min_, y_min_, x_max_, y_max_)
        img[:,:,2] = change_val(img[:,:,2], 0, len,  x_min_, y_min_, x_max_, y_max_)

    for bb in pred_boxes:
        x_min_, y_min_, x_max_, y_max_ = bb
        x_min_, y_min_, x_max_, y_max_ = int(x_min_),int( y_min_), int(x_max_), int(y_max_)
        img[:,:,0] = change_val(img[:,:,0], 0, len, x_min_, y_min_, x_max_, y_max_)
        img[:,:,1] = change_val(img[:,:,1], 255, len,  x_min_, y_min_, x_max_, y_max_)
        img[:,:,2] = change_val(img[:,:,2], 0, len,  x_min_, y_min_, x_max_, y_max_)

        max_iou = 0.0
        for gt_bb in gt_boxes:
            cur_iou = evaluate_box_JI(bb, gt_bb)
            # cur_iou = evaluate_box_dice(bb, gt_bb)
            if cur_iou > max_iou:
                max_iou = cur_iou

        text_loc = (x_min_, y_min_ - 10)
        img = cv2.putText(img.copy(), str(round(max_iou, 3)), text_loc,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img


def evaluate_box_JI(gt_box, pr_box):
    if pr_box[0] >= gt_box[2] or pr_box[2] <= gt_box[0] or pr_box[1] >= gt_box[3] or pr_box[3] <= gt_box[1]:
        return 0.0
    else:
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        pr_area = (pr_box[2] - pr_box[0]) * (pr_box[3] - pr_box[1])
        overlap_start = (max(gt_box[0], pr_box[0]), max(gt_box[1], pr_box[1]))
        overlap_end = (min(gt_box[2], pr_box[2]), min(gt_box[3], pr_box[3]))

        area_numerator = (overlap_end[1] - overlap_start[1]) * (overlap_end[0] - overlap_start[0])
        area_denominator = gt_area + pr_area - area_numerator
        overlap_ratio = area_numerator / area_denominator

    return overlap_ratio


def evaluate_box_dice(gt_box, pr_box):
    if pr_box[0] >= gt_box[2] or pr_box[2] <= gt_box[0] or pr_box[1] >= gt_box[3] or pr_box[3] <= gt_box[1]:
        return 0.0
    else:
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        pr_area = (pr_box[2] - pr_box[0]) * (pr_box[3] - pr_box[1])
        overlap_start = (max(gt_box[0], pr_box[0]), max(gt_box[1], pr_box[1]))
        overlap_end = (min(gt_box[2], pr_box[2]), min(gt_box[3], pr_box[3]))

        area_numerator = 2.0 * (overlap_end[1] - overlap_start[1]) * (overlap_end[0] - overlap_start[0])
        area_denominator = gt_area + pr_area
        overlap_ratio = area_numerator / area_denominator

    return overlap_ratio


# Save detected knee bbox for classification and segmentation
def save_pred_box_coors(save_dir, gt_boxes, gt_classes, pr_boxes, img_name, overlap_ratio=0.75):
    bone_dict = {"coors": [], "classes": []}
    for gt_box, gt_class in zip(gt_boxes, gt_classes):
        for ind, pr_box in enumerate(pr_boxes):
            if evaluate_box_JI(gt_box, pr_box) >= overlap_ratio:
                bone_dict["coors"].append(pr_box*8)
                bone_dict["classes"].append(gt_class)
    dd.io.save(os.path.join(save_dir, img_name+".h5"), bone_dict)


# # Overlay bbox and associated class
# def overlay_bbox_class(img, bboxes, classes=None, len=1, rgb=(255, 0, 0), pos="left"):
#     for ind, bb in enumerate(bboxes):
#     # for bb, clss in zip(bboxes, classes):
#         # Overlay bbox
#         x_min_, y_min_, x_max_, y_max_ = bb
#         x_min_, y_min_, x_max_, y_max_ = int(x_min_),int( y_min_), int(x_max_), int(y_max_)
#         img[:,:,0] = change_val(img[:,:,0], rgb[0], len, x_min_, y_min_, x_max_, y_max_)
#         img[:,:,1] = change_val(img[:,:,1], rgb[1], len,  x_min_, y_min_, x_max_, y_max_)
#         img[:,:,2] = change_val(img[:,:,2], rgb[2], len,  x_min_, y_min_, x_max_, y_max_)
#
#         if classes is not None:
#             # Overly class
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             if pos == "left":
#                 col_mid = int(bb[0] + (bb[2] - bb[0]) / 3.0)
#             elif pos == "right":
#                 col_mid = int(bb[0] + (bb[2] - bb[0]) * 2 / 3.0)
#             else:
#                 raise ValueError("Unknow position parameter")
#
#             row_mid = int((bb[3] + bb[1]) / 2.0)
#             # img = cv2.putText(img.copy(), str(clss), (col_mid, row_mid), font, 1, rgb, 2)
#             img = cv2.putText(img.copy(), str(classes[ind]), (col_mid, row_mid), font, 1, rgb, 2)
#
#     return img
#
#
# def overlay_gt_img(t_img, t_box, t_cls):
#     np_img = tensor_to_img(t_img)
#     bbox = t_box.squeeze_().numpy()
#     clss = t_cls.squeeze_().numpy()
#
#     overlay_img = overlay_bbox_class(
#         np_img, bbox, clss, len=6, rgb=(0, 255, 0), pos="left")
#     overlay_img = overlay_img.astype(np.uint8)
#
#     return overlay_img
