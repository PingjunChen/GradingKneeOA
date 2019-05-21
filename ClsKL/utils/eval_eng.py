# -*- coding: utf-8 -*-

import os, sys, pdb
import torch
from torch.autograd import Variable
import numpy as np
import itertools
import deepdish as dd
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .eval_util import ordinal_mse
from .layer_util import extract_gap_layer, extract_vgg_fea_layer
from .layer_util import gen_cam_visual
from .grad_cam import GradCam, show_cam_on_image


def eval_test(args, model, dset_loaders, dset_size, phase="test"):
    labels_all = [] * dset_size[phase]
    preds_all = [] * dset_size[phase]

    for data in dset_loaders[phase]:
        inputs, labels, _ = data
        inputs = Variable(inputs.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        labels_np = labels.numpy()
        labels = labels_np.tolist()
        labels_all.extend(labels)
        preds_cpu = preds.cpu()
        preds_np = preds_cpu.numpy()
        preds = preds_np.tolist()
        preds_all.extend(preds)

    conf_matrix = confusion_matrix(labels_all, preds_all)
    acc = 1.0*np.trace(conf_matrix)/np.sum(conf_matrix)
    mse = ordinal_mse(conf_matrix)

    return acc, mse


def gen_vis_loc(args, phase, dset_loaders, dset_size, save_dir):
    model = torch.load(args.best_model_path)
    model.cuda()
    model.eval()

    count, ttl_num = 0, dset_size[phase]
    for data in dset_loaders[phase]:
        inputs, labels, paths = data
        count += len(paths)
        print("Processing {}/{}".format(count, ttl_num))
        inputs = Variable(inputs.cuda())

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
    model.cuda()
    model.eval()

    labels_all = [] * dset_size[phase]
    preds_all = [] * dset_size[phase]
    feas_all = []

    for data in dset_loaders[phase]:
        inputs, labels, paths = data
        inputs = Variable(inputs.cuda())

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        # retrieve gap layer
        # gaps = extract_gap_layer(model, inputs)
        vgg_feas = extract_vgg_fea_layer(model, inputs)
        feas_all.append(vgg_feas.data.cpu().numpy())

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
    print('True/Total: {}/{}'.format(np.trace(conf_matrix), np.sum(conf_matrix)))
    # print('MSE: {:.4f}'.format(ordinal_mse(conf_matrix, poly_num=2)))
    print('Acc: {:.3f} ABE: {:.3f}'.format(acc, ordinal_mse(conf_matrix, poly_num=1)))

    # plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('Acc: {:.3f} MAE: {:.3f}'.format(acc, ordinal_mse(conf_matrix, poly_num=1)), fontsize=18)
    # classes = [0, 1, 2, 3, 4]
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
    # plt.yticks(tick_marks, classes, fontsize=14)
    # for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    #     plt.text(j, i, format(conf_matrix[i, j], 'd'),
    #              horizontalalignment="center",
    #              color="white" if conf_matrix[i, j] > conf_matrix.max() / 2. else "black")
    # plt.tight_layout()
    # plt.savefig('cm04.svg')



def gen_grad_cam(args, phase, dset_loaders, dset_size, save_dir):
    model = torch.load(args.best_model_path)
    model.cuda()
    model.eval()

    alpha = 0.6
    beta  = 1 - alpha
    pixel_mean, pixel_std = 0.66133188,  0.21229856

    grad_cam = GradCam(model, target_layer_names = ["35"], use_cuda=1)
    count, ttl_num = 0, dset_size[phase]
    for data in dset_loaders[phase]:
        inputs, labels, paths = data
        count += len(paths)
        print("Processing {}/{}".format(count, ttl_num))
        inputs = Variable(inputs.cuda())

        for input, label, path in zip(inputs, labels, paths):
            input.unsqueeze_(0)
            target_index = label.tolist()
            mask = grad_cam(input, target_index)

            input = input.permute(0, 2, 3, 1)
            img = input.data.cpu().numpy()
            img = (np.squeeze(img) * pixel_std) + pixel_mean

            cam = (mask * 255.0).astype(np.uint8)
            cam_rgb = cv2.applyColorMap(cam, cv2.COLORMAP_HSV)

            img = (img * 255.0).astype(np.uint8)
            img_cam = cv2.addWeighted(img, alpha, cam_rgb, beta, 0)

            save_path = os.path.join(save_dir, str(target_index), path)
            cv2.imwrite(save_path, img_cam)
