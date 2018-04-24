# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt
import torch
import deepdish as dd


from .utils.timer import Timer
from .utils import yolo as yolo_utils
from .utils.cython_yolo import yolo_to_bbox
from .proj_utils.plot_utils  import plot_scalar
from .proj_utils.torch_utils import to_device
from .proj_utils.local_utils import writeImg, mkdirs
from .proj_utils.local_utils import overlay_bbox, overlay_bbox_class
from .proj_utils.local_utils import overlay_bbox_IoU

def kl_pred(cls, img, cfg=None):
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


def tensor_to_img(t_img):
    t_img.mul_(0.5).add_(0.5)
    t_img.squeeze_()
    np_img = t_img.numpy().transpose(1, 2, 0)
    np_img = (np_img * 255.0).astype(np.uint8)

    return np_img


def overlay_gt_img(t_img, t_box, t_cls):
    np_img = tensor_to_img(t_img)
    bbox = t_box.squeeze_().numpy()
    clss = t_cls.squeeze_().numpy()

    overlay_img = overlay_bbox_class(
        np_img, bbox, clss, len=6, rgb=(0, 255, 0), pos="left")
    overlay_img = overlay_img.astype(np.uint8)

    return overlay_img

def evaluate_box_overlap(gt_box, pr_box):
    if pr_box[0] >= gt_box[2] or pr_box[2] <= gt_box[0] or pr_box[1] >= gt_box[3] or pr_box[3] <= gt_box[1]:
        return 0.0
    else:
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        pr_area = (pr_box[2] - pr_box[0]) * (pr_box[3] - pr_box[1])
        overlap_start = (max(gt_box[0], pr_box[0]), max(gt_box[1], pr_box[1]))
        overlap_end = (min(gt_box[2], pr_box[2]), min(gt_box[3], pr_box[3]))
        overlap_area = (overlap_end[1] - overlap_start[1]) * (overlap_end[0] - overlap_start[0])
        overlap_ratio = overlap_area * 2.0 / (gt_area + pr_area)

    return overlap_ratio

def save_pred_box_coors(save_dir, gt_boxes, gt_classes, pr_boxes, img_name, overlap_ratio=0.75):
    bone_dict = {"coors": [], "classes": []}
    for gt_box, gt_class in zip(gt_boxes, gt_classes):
        for ind, pr_box in enumerate(pr_boxes):
            if evaluate_box_overlap(gt_box, pr_box) >= overlap_ratio:
                bone_dict["coors"].append(pr_box*8)
                bone_dict["classes"].append(gt_class)
    dd.io.save(os.path.join(save_dir, img_name+".h5"), bone_dict)


def evaluate_classification(gt_boxes, gt_classes, pr_boxes, pr_classes, num=5, overlap_ratio=0.75):
    t_box_num = 0
    all_box_num = len(pr_boxes)
    pred_matrix = np.zeros((num, num), dtype=np.int)
    overlap_list = []

    for gt_box, gt_class in zip(gt_boxes, gt_classes):
        for pr_box, pr_class in zip(pr_boxes, pr_classes):
            cur_overlap = evaluate_box_overlap(gt_box, pr_box)
            if cur_overlap >= overlap_ratio:
                pred_matrix[gt_class, pr_class] += 1
                t_box_num += 1
                overlap_list.append(cur_overlap)

    return pred_matrix, t_box_num, all_box_num, overlap_list


def test_eng(dataloader, model_root, save_root, net, args, cfg):
    net.eval()
    weightspath = os.path.join(model_root, args.model_name)
    weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
    print("===="*20)
    print('Model name is {}'.format(weightspath))
    net.load_state_dict(weights_dict)
    print("Num images: {}".format(len(dataloader)))
    _t = {'im_detect': Timer(), 'misc': Timer()}

    evalate_matrix = np.zeros((5, 5), dtype=np.int)
    true_box_num, total_box_num = 0, 0
    all_overlap = []
    # import pdb; pdb.set_trace()
    for ind, data in enumerate(dataloader):
        cur_img, cur_boxes, cur_classes, cur_name = data
        # detection
        _t['im_detect'].tic()
        result_dict = kl_pred(net, cur_img, cfg=cfg)
        bbox_pred, iou_pred, prob_pred  = result_dict['bbox'], result_dict['iou'], result_dict['prob']
        detect_time = _t['im_detect'].toc()
        # postprocessing
        _t['misc'].tic()
        # bboxes, scores, cls_inds = yolo_utils.postprocess_bbox(bbox_pred, iou_pred, prob_pred, cur_img.shape[2:], cfg, thresh=0.25)
        bboxes, scores, cls_inds = yolo_utils.postprocess_bbox(bbox_pred, iou_pred, prob_pred, cur_img.shape[2:], cfg, thresh=0.12)
        utils_time = _t['misc'].toc()

        gt_boxes = cur_boxes.squeeze().numpy()
        gt_classes = cur_classes.squeeze().numpy()
        cls_mat, true_num, total_num, overlap_list = evaluate_classification(gt_boxes, gt_classes, bboxes, cls_inds)
        if total_num != 2: # Check wrong detection file
            print("Name: {}, num: {}".format(cur_name[0], total_num))
        evalate_matrix += cls_mat
        true_box_num += true_num
        total_box_num += total_num
        all_overlap.extend(overlap_list)

        # save_pred_box_coors(save_root, gt_boxes, gt_classes, bboxes, cur_name[0])

        if (ind+1) % 100 == 0:
            print('{}/{} detection time {:.4f}, post_processing time {:.4f}'.format(
                ind+1, len(dataloader), detect_time, utils_time))

        # Overlay gt box and prediction box and overlap value

        gt_boxes = cur_boxes.squeeze_().numpy()
        overlaid_img = overlay_bbox_IoU(tensor_to_img(cur_img), bboxes, gt_boxes)
        writeImg(overlaid_img, os.path.join(save_root, cur_name[0]+'.png'))

        # # Overlay and save
        # # ori_overlay = overlay_gt_img(cur_img, cur_boxes, cur_classes)
        # ori_overlay = tensor_to_img(cur_img)
        # # overlaid_img = overlay_bbox_class(ori_overlay, bboxes, cls_inds, len=6, rgb=(255, 0, 0), pos="right")
        # overlaid_img = overlay_bbox_class(ori_overlay, bboxes, len=6, rgb=(255, 0, 0))
        # writeImg(overlaid_img, os.path.join(save_root, cur_name[0]+'.png'))

    print("---Detection accuracy---")
    print("Number of object: {}, Mean IoU is: {}".format(len(all_overlap), np.mean(all_overlap)))
    print("True predicted knee: {}\n All predicted knee: {}\n Total accuracy: {:.4f}\n".format(
        true_box_num, total_box_num, true_box_num*1.0/total_box_num))
