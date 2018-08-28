# -*- coding: utf-8 -*-

import os, sys, pdb

import torch
import torch.nn as nn
import numpy as np
from skimage import io, transform


def extract_vgg_fea_layer(model, inputs):
    x = model.features(inputs)
    x = x.view(x.size(0), -1)
    fea_extractor = nn.Sequential(*list(model.classifier.children())[:-1])
    vgg_feas = fea_extractor(x)

    return vgg_feas

def extract_gap_layer(model, inputs):
    x = model.conv1(inputs)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    gap = model.avgpool(x)
    x = x.view(gap.size(0), -1)

    return x

def gen_cam_visual(model, inputs):
    x = model.conv1(inputs)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    fea77 = model.layer4(x)
    gap512 = model.avgpool(fea77)
    gap512 = gap512.view(gap512.size(0), -1)
    pred_prob = model.fc(gap512)

    # Get prediction result
    _, pred_ind = torch.max(pred_prob, dim=1)
    preds = pred_ind.data
    fc_weight = model.fc.weight.data
    class_w = fc_weight[preds, :]

    class_w = class_w.cpu().numpy()
    fea77 = fea77.data.cpu().numpy()

    cams = []
    for ind in range(fea77.shape[0]):
        cur_w = class_w[ind:ind+1, :]
        cur_fea = fea77[ind].reshape(fea77.shape[1], -1)
        fea_map = np.matmul(cur_w, cur_fea).reshape(fea77.shape[2], fea77.shape[3])
        fea_map = (fea_map - np.amin(fea_map)) * 1.0 / (np.amax(fea_map) - np.amin(fea_map))
        fea_map = transform.resize(fea_map, (inputs.shape[2], inputs.shape[3]))

        cams.append(fea_map)

    preds = preds.cpu().numpy()
    cams = np.array(cams)

    return preds, cams
