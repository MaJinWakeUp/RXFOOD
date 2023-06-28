#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:44:50 2022

@author: majin
"""
import os
import random
import numpy as np

def defacto(root_dir):
    manis = ['splicing', 'copymove', 'inpainting']
        
    mani_paths = []
    # manipulation images
    for m in manis:
        paths = []
        imgs_path = os.path.join(root_dir, m+'_archive', m+'_img', 'img')
        masks_path = os.path.join(root_dir, m+'_archive', m+'_annotations', 'probe_mask')
        
        imgs_list = os.listdir(imgs_path)
        masks_list = os.listdir(masks_path)
        for f in imgs_list:
            if f in masks_list:
                rgb_path = os.path.join(imgs_path, f)
                mask_path = os.path.join(masks_path, f)
                paths.append((rgb_path, mask_path, "1"))
            if f.replace('.tif','.jpg') in masks_list:
                rgb_path = os.path.join(imgs_path, f)
                mask_path = os.path.join(masks_path, f.replace('.tif','.jpg'))
                paths.append((rgb_path, mask_path, "1"))
        
        mani_paths.append(paths)
    for i in range(3):
        random.shuffle(mani_paths[i])
    # print(len(mani_paths))
    
    auth_paths = []
    auth_folder = '/home/majin/datasets/COCO-2017/train2017'
    auimgs_list = os.listdir(auth_folder)
    for f in auimgs_list:
        if f.endswith('.jpg'):
            rgb_path = os.path.join(auth_folder, f)
            auth_paths.append((rgb_path, "NULL", "0"))
    random.shuffle(auth_paths)
    
    train_file = os.path.join(root_dir, 'train.txt')
    val_file = os.path.join(root_dir, 'val.txt')
    test_file = os.path.join(root_dir, 'test.txt')
    with open(train_file,'w') as f:
        for i in np.arange(33000):
            f.write(' '.join(mani_paths[0][i])+'\n')
        for i in np.arange(15000):
            f.write(' '.join(mani_paths[1][i])+'\n')
        for i in np.arange(16000):
            f.write(' '.join(mani_paths[2][i])+'\n')
        for i in np.arange(20000):
            f.write(' '.join(auth_paths[i])+'\n')
    with open(val_file,'w') as f:
        for i in np.arange(33000,34000):
            f.write(' '.join(mani_paths[0][i])+'\n')
        for i in np.arange(15000,16000):
            f.write(' '.join(mani_paths[1][i])+'\n')
        for i in np.arange(16000,17000):
            f.write(' '.join(mani_paths[2][i])+'\n')
        for i in np.arange(20000,21000):
            f.write(' '.join(auth_paths[i])+'\n')
    with open(test_file,'w') as f:
        for i in np.arange(34000,36000):
            f.write(' '.join(mani_paths[0][i])+'\n')
        for i in np.arange(16000,18000):
            f.write(' '.join(mani_paths[1][i])+'\n')
        for i in np.arange(17000,19000):
            f.write(' '.join(mani_paths[2][i])+'\n')
        for i in np.arange(21000,27000):
            f.write(' '.join(auth_paths[i])+'\n')
    return

def nist16(root_dir):
    records = []
    import pandas as pd
    ref_files = [os.path.join(root_dir,'reference', x, 'NC2016-'+x+'-ref.csv')
                 for x in ['manipulation', 'remove', 'splice']]
    img_sets = set()
    for file in ref_files:
        data = pd.read_csv(file, sep='|', usecols=['ProbeFileName', 'ProbeMaskFileName'])
        num = data.shape[0]
        for i in range(num):
            if isinstance(data['ProbeMaskFileName'][i], str):
                rgb_path = os.path.join(root_dir, data['ProbeFileName'][i])
                mask_path = os.path.join(root_dir, data['ProbeMaskFileName'][i])
                if rgb_path not in img_sets:
                    img_sets.add(rgb_path)
                    records.append((rgb_path, mask_path, "1"))
            else:
                continue
    
    train_file = os.path.join(root_dir, 'train.txt')
    test_file = os.path.join(root_dir, 'test.txt')
    with open(train_file,'w') as f:
        for i in np.arange(404):
            f.write(' '.join(records[i])+'\n')
    with open(test_file,'w') as f:
        for i in np.arange(404,564):
            f.write(' '.join(records[i])+'\n')
    return

def coverage(root_dir):
    rgb_folder = os.path.join(root_dir, 'image')
    mask_folder = os.path.join(root_dir, 'mask')
    rgb_list = os.listdir(rgb_folder)
    
    paths = []
    for file in rgb_list:
        if file.endswith('t.tif'):
            rgb_path = os.path.join(rgb_folder, file)
            mask_name = file.replace('t.tif','forged.tif')
            mask_path = os.path.join(mask_folder, mask_name)
            paths.append((rgb_path, mask_path, "1"))
        else:
            continue
    train_file = os.path.join(root_dir, 'train.txt')
    test_file = os.path.join(root_dir, 'test.txt')
    with open(train_file,'w') as f:
        for i in np.arange(75):
            f.write(' '.join(paths[i])+'\n')
    with open(test_file,'w') as f:
        for i in np.arange(75,100):
            f.write(' '.join(paths[i])+'\n')
    return

def columbia(root_dir):
    rgb_folder = os.path.join(root_dir,'4cam_splc')
    mask_folder = os.path.join(root_dir, '4cam_splc', 'edgemask')
    rgb_list = os.listdir(rgb_folder)
    
    paths = []
    for file in rgb_list:
        if file.endswith('.tif'):
            rgb_path = os.path.join(rgb_folder, file)
            mask_name = file.split('.')[0]+'_edgemask.jpg'
            mask_path = os.path.join(mask_folder, mask_name)
            paths.append((rgb_path, mask_path, "1"))
        else:
            continue

    test_file = os.path.join(root_dir, 'test.txt')
    with open(test_file,'w') as f:
        for i in range(len(paths)):
            f.write(' '.join(paths[i])+'\n')
    return

def casia1(root_dir):
    rgb_folder_cm = os.path.join(root_dir, 'CASIA1.0dataset', 'ModifiedTp', 'CM')
    rgb_folder_sp = os.path.join(root_dir, 'CASIA1.0dataset', 'ModifiedTp', 'Sp')
    mask_folder_cm = os.path.join(root_dir, 'CASIA1.0groundtruth', 'CM')
    mask_folder_sp = os.path.join(root_dir, 'CASIA1.0groundtruth', 'Sp')
    
    paths = []
    mask_files_cm = os.listdir(mask_folder_cm)
    for file in mask_files_cm:
        mask_path = os.path.join(mask_folder_cm, file)
        rgb_name = file.replace('_gt.png', '.jpg')
        rgb_path = os.path.join(rgb_folder_cm, rgb_name)
        paths.append((rgb_path, mask_path, "1"))
    mask_files_sp = os.listdir(mask_folder_sp)
    for file in mask_files_sp:
        mask_path = os.path.join(mask_folder_sp, file)
        rgb_name = file.replace('_gt.png', '.jpg')
        rgb_path = os.path.join(rgb_folder_sp, rgb_name)
        paths.append((rgb_path, mask_path, "1"))
    test_file = os.path.join(root_dir, 'test.txt')
    with open(test_file,'w') as f:
        for i in range(len(paths)):
            f.write(' '.join(paths[i])+'\n')
    return

def casia2(root_dir):
    names_file = os.path.join(root_dir, 'CASIA2.0_revised','tp_list.txt')
        
    paths = []
    with open(names_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        base_name = line.strip()
        rgb_path = os.path.join(root_dir, 'CASIA2.0_revised', 'Tp', base_name)
        mask_name = base_name.split('.')[0]+'_gt.png'
        mask_path = os.path.join(root_dir, 'CASIA2.0_Groundtruth', mask_name)
        paths.append((rgb_path, mask_path,"1"))
    
    train_file = os.path.join(root_dir, 'train.txt')
    #test_file = os.path.join(root_dir, 'test.txt')
    with open(train_file,'w') as f:
        for i in np.arange(len(paths)):
            f.write(' '.join(paths[i])+'\n')
    return

if __name__ == "__main__":
    # defacto_root = '/home/majin/datasets/Defacto'
    # defacto(defacto_root)
    
    # nist16_root = '/home/majin/datasets/NIST16/NC2016_Test0613'
    # nist16(nist16_root)
    
    # coverage_root = '/home/majin/datasets/COVERAGE'
    # coverage(coverage_root)
    
    # columbia_root = '/home/majin/datasets/Columbia-uncompressed'
    # columbia(columbia_root)
    
    # casia1_root = '/home/majin/datasets/CASIA1.0'
    # casia1(casia1_root)
    
    casia2_root = '/home/majin/datasets/CASIA2.0'
    casia2(casia2_root)