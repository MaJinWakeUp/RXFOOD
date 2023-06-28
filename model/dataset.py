#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 16:49:54 2021

@author: majin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:21:01 2021

@author: majin
"""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image


class MSSODDataset(Dataset):
    def __init__(self, root_dir, index_file, transforms=None):
        super().__init__()
        self.root_dir = root_dir
        self.records = self.get_records(index_file)
        self.transforms = transforms

    def get_records(self, index_file):
        with open(index_file, 'r') as f:
            content = f.readlines()
        records = []
        for line in content:
            x = line.strip()
            if x.split('_')[-1] == 'rgb':
                records.append(tuple([x+'.png', x.replace('rgb', 'nir')+'.png', x+'.png']))
        return records
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rgb_name, nir_name, gt_name = self.records[idx]
        rgb_path = os.path.join(self.root_dir, 'images', rgb_name)
        nir_path = os.path.join(self.root_dir, 'images', nir_name)
        gt_path = os.path.join(self.root_dir, 'gt', gt_name)
        rgb = Image.open(rgb_path)
        nir = Image.open(nir_path)
        gt = Image.open(gt_path).convert('L')
        if self.transforms is not None:
            rgb = self.transforms(rgb)
            nir = self.transforms(nir)
            gt = self.transforms(gt)
        return rgb, nir, gt

import torchvision.transforms as T
class NYUv2(Dataset):
    """NYUv2 dataset
    
    Args:
        root (string): Root directory path.
        split (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
        target_type (string, optional): Type of target to use, ``semantic``, ``depth`` or ``normal``. 
        num_classes (int, optional): The number of classes, must be 40 or 13. Default:13.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry and returns a transformed version.
    """
    def __init__(self,
                 root,
                 split='train',
                 num_classes=40,
                 transforms=None,):
        super().__init__()
        assert(split in ('train', 'test'))
        self.transforms = transforms
        self.root = root
        self.split = split
        self.num_classes = num_classes
        
        img_names = os.listdir( os.path.join(self.root, 'image', self.split) )
        img_names.sort()
        images_dir = os.path.join(self.root, 'image', self.split)
        self.images = [os.path.join(images_dir, name) for name in img_names]
        
        depth_dir = os.path.join(self.root, 'depth', self.split)
        self.depths = [os.path.join(depth_dir, name) for name in img_names]
        self._depth_mean = 2841.94941272766
        self._depth_std = 1417.2594281672277

        semantic_dir = os.path.join(self.root, 'seg%d'%self.num_classes, self.split)
        self.targets = [os.path.join(semantic_dir, name) for name in img_names]
        
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        depth = Image.open(self.depths[idx])
        target = Image.open(self.targets[idx])
        image = T.ToTensor()(image)
        depth = T.ToTensor()(depth).float()
        # pre-process depth
        # depth = (depth - self._depth_mean) / self._depth_std
        depth = depth / 1000.0
        # depth = depth.clamp(0, 5)
        target = np.asarray(target)
        target[target==255] = self.num_classes
        target = torch.from_numpy(target).long()
        if self.transforms is not None:
            image = self.transforms(image)
            depth = self.transforms(depth)
            target = self.transforms(target.unsqueeze(0))
        return image, depth, target.squeeze()

    def __len__(self):
        return len(self.images)

class ReDWeb(Dataset):
    def __init__(self,
                 root,
                 split='train',
                 transforms=None,
                 rgb_transform=None,
                 depth_transform=None,
                 ):
        super().__init__()
        assert(split in ('train', 'test'))
        self.transforms = transforms
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.root = root
        self.split = split
        
        rgb_dir = os.path.join(self.root, self.split+'set', 'RGB')
        img_names = os.listdir(rgb_dir)
        img_names.sort()
        self.images = [os.path.join(rgb_dir, name) for name in img_names]
        
        depth_dir = os.path.join(self.root, self.split+'set', 'depth')
        self.depths = [os.path.join(depth_dir, name.replace('.jpg', '.png')) for name in img_names]

        target_dir = os.path.join(self.root, self.split+'set', 'GT')
        self.targets = [os.path.join(target_dir, name.replace('.jpg', '.png')) for name in img_names]
        
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        depth = Image.open(self.depths[idx]).convert('L')
        target = Image.open(self.targets[idx]).convert('L')
        
        if self.transforms is not None:
            image = self.transforms(image)
            depth = self.transforms(depth)
            target = self.transforms(target)
        if self.rgb_transform is not None:
            image = self.rgb_transform(image)
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)
        return image, depth, target

    def __len__(self):
        return len(self.images)



if __name__=='__main__':
    from torchvision import transforms
    TransForms = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Resize([256, 256]),])
    root_dir = '/home/majin/datasets/NYUv2/nyuv2-python-toolkit/NYUv2'
    dataset = NYUv2(root_dir, )