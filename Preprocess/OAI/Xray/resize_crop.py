# coding: utf-8

import os, sys, pdb
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt
from skimage import transform, io
import math
import dicom


def resize_crop_save(dcm_filepath, save_dir, new_ratio=0.14, crop_height=2048, crop_width=2560):
    # load and transform to [0, 1]
    bone_dcm = dicom.read_file(dcm_filepath)
    # bone_arr = bone_dcm.pixel_array
    # bone_arr_reg = bone_arr/2**12

    # Resize to have
    if 'PixelSpacing' in bone_dcm:
        # ratio = bone_dcm.PixelSpacing
        return True
    else:
        print("No pixle spacing info for {}.".format(
            os.path.basename(dcm_filepath)))
        return False

    # resolu_size = bone_arr.shape
    # new_resolu = np.array(resolu_size)*np.array(ratio)/new_ratio
    # new_resolu_h = int(round(new_resolu[0]))
    # new_resolu_w = int(round(new_resolu[1]))
    # resized_img = transform.resize(bone_arr,(new_resolu_h, new_resolu_w))
    #
    # # Crop image
    # start_height = int(round(new_resolu_h/2 - crop_height/2))
    # start_width = int(round(new_resolu_w/2 - crop_width/2))
    # end_height = start_height + crop_height
    # end_width = start_width + crop_width
    # if start_height < 0 or start_width < 0 or \
    #     end_height > new_resolu_h or end_width > new_resolu_w:
    #     print("Image too small, not able to crop {}.".format(
    #         os.path.basename(dcm_filepath)))
    #     return False
    # res_crop = resized_img[start_height:end_height, start_width:end_width]
    #
    # # Save Image
    # save_imgname = os.path.basename(dcm_filepath).replace('dcm', 'png')
    # save_img_path = os.path.join(save_dir, save_imgname)
    # imsave(save_img_path, res_crop)
    #
    # return True


def batch_preprocess(root_dir, process_dir):
    image_list = os.listdir(root_dir)
    print("There are {} images in total".format(len(image_list)))
    processed_num = 0
    for ind, cur_img_name in enumerate(image_list):
        print("Processing  {}, {}/{}".format(cur_img_name, ind+1, len(image_list)))
        status = resize_crop_save(os.path.join(root_dir, cur_img_name), process_dir)
        if status == True:
            processed_num += 1

    print("There are {} images successfully processed".format(processed_num))

if __name__ == '__main__':
    np.random.seed(1234)
    root_dir = '../../../data/XrayLong/BL'
    process_dir = '../../../data/ImgsBL/XrayBLimg'

    batch_preprocess(root_dir, process_dir)
