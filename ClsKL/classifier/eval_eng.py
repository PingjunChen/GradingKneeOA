# -*- coding: utf-8 -*-

import os, sys, pdb
import torch
from torch.autograd import Variable
import numpy as np
import deepdish as dd
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from eval_util import ordinal_mse
from layer_util import extract_gap_layer
from layer_util import gen_cam_visual
# from pydaily.plots import subplot


def gen_vis_loc(args, phase, dset_loaders, dset_size, save_dir):
    model = torch.load(args.best_model_path)
    model.cuda(args.cuda_id)
    model.eval()

    count, ttl_num = 0, dset_size[phase]
    for data in dset_loaders[phase]:
        inputs, labels, paths = data
        count += len(paths)
        print("Processing {}/{}".format(count, ttl_num))
        if args.cuda_id >= 0:
            inputs = Variable(inputs.cuda(args.cuda_id))
        else:
            inputs = Variable(inputs)

        # Prediction & CAMs
        preds, cams = gen_cam_visual(model, inputs)

        # Inputs & Labels
        labels = labels.numpy()
        pixel_mean, pixel_std = 0.66133188,  0.21229856
        inputs = inputs.permute(0, 2, 3, 1)
        inputs = inputs.data.cpu().numpy()
        inputs = (inputs * pixel_std) + pixel_mean
        inputs = np.clip(inputs, 0.0, 1.0)

        # inputs / cam_list / labels /  preds
        for img, label, cam, pred, path in zip(inputs, labels, cams, preds, paths):
            alpha = 0.6
            beta  = 1 - alpha
            cam = (cam * 255.0).astype(np.uint8)
            cam_rgb = cv2.applyColorMap(cam, cv2.COLORMAP_HSV)
            img = (img * 255.0).astype(np.uint8)
            img_cam = cv2.addWeighted(img, alpha, cam_rgb, beta, 0)

            # Plotting
            l_title = "Input: (Grade " + str(label) + ")"
            r_title = "Heatmap: (Grade " + str(pred) + ")"
            suptitle = path + ":" + str(label==pred)
            #subplot.plot_lr(img, img_cam, l_title, r_title, suptitle)

            # Saving plotting
            save_folder = os.path.join(save_dir, str(label))
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            save_path = os.path.join(save_folder, os.path.splitext(path)[0] + ".png")
            # save_path = os.path.join(save_folder, os.path.splitext(path)[0] + ".pdf")
            # with PdfPages(save_path) as pdf:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
            axes[0].imshow(img)
            axes[0].set_title(l_title)
            axes[1].imshow(img_cam)
            axes[1].set_title(r_title)
            plt.suptitle(suptitle, fontsize=16)
            plt.savefig(save_path)
            plt.close('all')

def eval_model(args, phase, dset_loaders, dset_size):
    model = torch.load(args.best_model_path)
    model.cuda(args.cuda_id)
    model.eval()

    labels_all = [] * dset_size[phase]
    preds_all = [] * dset_size[phase]
    feas_all = []

    for data in dset_loaders[phase]:
        inputs, labels, paths = data
        if args.cuda_id >= 0:
            inputs = Variable(inputs.cuda(args.cuda_id))
        else:
            inputs = Variable(inputs)
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        # # retrieve gap layer
        # gaps = extract_gap_layer(model, inputs)
        # feas_all.append(gaps.data.cpu().numpy())

        labels_np = labels.numpy()
        labels = labels_np.tolist()
        labels_all.extend(labels)
        preds_cpu = preds.cpu()
        preds_np = preds_cpu.numpy()
        preds = preds_np.tolist()
        preds_all.extend(preds)

    conf_matrix = confusion_matrix(labels_all, preds_all)
    print("In {}: confusion matrix is:\n {}".format(phase, conf_matrix))
    acc = 1.0*np.trace(conf_matrix)/np.sum(conf_matrix)
    print('Acc: {:.4f}  True/Total: {}/{}'.format(
        acc, np.trace(conf_matrix), np.sum(conf_matrix)))
    print('MSE: {:.4f}'.format(ordinal_mse(conf_matrix, poly_num=2)))
    print('ABE: {:.4f}'.format(ordinal_mse(conf_matrix, poly_num=1)))

    # # save features for tsne
    # feas_all = np.concatenate(feas_all)
    # dd.io.save('feas1646_auto.h5', {'data': feas_all, 'target': labels_all})
