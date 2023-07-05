#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 09:18:37 2021

@author: majin
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import cv2
# torch.cuda.set_device(0)
import argparse
import numpy as np
from model.dataset import MSSODDataset
from PIL import Image
from model.net import (DualBASNet, DualBASNetwithRXFOOD,
                        DualCPD, DualCPDwithRXFOOD)
import datetime
import torch.nn.functional as F
import torchvision.transforms as T
from skimage import io

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(os.path.join(d_dir, imidx+'.png'))
    imidx_ = imidx.replace('rgb', 'nir')
    imo.save(os.path.join(d_dir, imidx_+'.png'))

def load_model(checkpoint):
    model = DualCPDwithRXFOOD()
    ret = model.load_state_dict(torch.load(checkpoint), strict=True)
    print(f'[Info] loaded weights: {checkpoint}')
    model = model.cuda()
    model.eval()
    return model

def inference(model, dataset, data_dir, save_dir, model_name):
    save_images_dir = os.path.join(save_dir, 'images')
    save_gt_dir = os.path.join(save_dir, 'gt')
    if not os.path.exists(save_images_dir):
        os.makedirs(save_images_dir)
    if not os.path.exists(save_gt_dir):
        os.makedirs(save_gt_dir)
    times = []
    for i in range(len(dataset)):
        print(i)
        with torch.no_grad():
            rgb, nir, gt = dataset[i]
            image_name1 = dataset.records[i][0]
            image_name2 = dataset.records[i][1]
            gt_name1 = dataset.records[i][2]
            gt_name2 = (dataset.records[i][2]).replace('rgb', 'nir')
            image_name1 = os.path.join(data_dir, 'images', image_name1)
            image_name2 = os.path.join(data_dir, 'images', image_name2)
            gt_name1 = os.path.join(data_dir, 'gt', gt_name1)
            gt_name2 = os.path.join(data_dir, 'gt', gt_name2)
            if not os.path.exists(os.path.join(save_images_dir, image_name1.split('/')[-1])):
                # read image and save
                image1 = cv2.imread(image_name1)
                cv2.imwrite(os.path.join(save_images_dir, image_name1.split('/')[-1]), image1)
                image2 = cv2.imread(image_name2)
                cv2.imwrite(os.path.join(save_images_dir, image_name2.split('/')[-1]), image2)
                gt1 = cv2.imread(gt_name1)
                cv2.imwrite(os.path.join(save_gt_dir, gt_name1.split('/')[-1]), gt1)
                gt2 = cv2.imread(gt_name2)
                cv2.imwrite(os.path.join(save_gt_dir, gt_name2.split('/')[-1]), gt2)
            
            rgb = rgb.unsqueeze(0).cuda()
            nir = nir.unsqueeze(0).cuda()
            gt = gt.unsqueeze(0).cuda()
            
            t0 = datetime.datetime.now()
            output, loss_list = model(rgb, nir, gt)
            t1 = datetime.datetime.now()
            # print(f'[Info] inference time: {(t1-t0).total_seconds()}s')
            times.append((t1-t0).total_seconds())
            predict = output[:,0,:,:]
            # predict = normPRED(predict)
            predict = F.sigmoid(predict)
            save_output(image_name1, predict, os.path.join(save_dir, model_name))
    print(f'[Info] average inference time: {np.mean(times)}s')
            

if __name__ == '__main__':
    torch.cuda.empty_cache()
    model_name = 'DualCPDwithRXFOOD_MSSOD'
    data_dir = '/home/majin/datasets/MultiSpectralSOD/Images'
    checkpoint = './log/'+model_name+'.pth'
    save_dir = os.path.join('./results/MSSOD/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, model_name)):
        os.makedirs(os.path.join(save_dir, model_name))
    model = load_model(checkpoint)
    TransForms = T.Compose([T.Resize((256, 256)),
                            T.ToTensor()])
    test_file = os.path.join(data_dir, 'test.txt')
    test_set = MSSODDataset(data_dir, test_file, TransForms)
        
    inference(model, test_set, data_dir, save_dir, model_name)