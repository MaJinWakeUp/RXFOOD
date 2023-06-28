#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:54:54 2021

@author: majin
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import argparse
import datetime
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from model.net import (DualBASNet, DualBASNetwithRXFOOD,
                       DualCPD, DualCPDwithRXFOOD)
from model.dataset import MSSODDataset
from utils.plotter import Plotter
from tqdm import tqdm
from utils.metrics import pix_acc, auc_roc, f1_score

def train(opt):
    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)
    if not os.path.exists(opt.graph_path):
        os.makedirs(opt.graph_path)
    
    TransForms = T.Compose([
                            T.ToTensor(),
                            T.Resize([256, 256]),
                            ])
    train_file = os.path.join(opt.data_dir, 'train.txt')
    train_set = MSSODDataset(opt.data_dir, train_file, TransForms)
    val_file = os.path.join(opt.data_dir, 'val.txt')
    val_set = MSSODDataset(opt.data_dir, val_file, TransForms)
    test_file = os.path.join(opt.data_dir, 'test.txt')
    test_set = MSSODDataset(opt.data_dir, test_file, TransForms)
    
    print(f'[Info] Training data: {len(train_set)}; Validation data: {len(val_set)}; Test data: {len(test_set)}')
    params = {'batch_size': opt.batch_size,
              'shuffle': True,
              'num_workers': opt.num_workers}
    
    train_generator = DataLoader(train_set, **params)
    val_generator = DataLoader(val_set, **params)
    test_generator = DataLoader(test_set, **params)
    
    # build network
    model = DualCPDwithRXFOOD()
    
    if opt.resume is not None:
        try:
            ret = model.load_state_dict(torch.load(opt.resume), strict=False)
            print(f'[Info] loaded weights: {os.path.basename(opt.resume)}')
        except:
            print('[Warning] failed to load weights, use initialized weights.')
            pass
    model = model.cuda()

    # train strategy
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1,
                                                     min_lr=1e-6, verbose=True)
    plotter_loss = Plotter(phases=['train','val'], metrics=['loss'], 
                           filename=os.path.join(opt.graph_path,f'loss_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.jpg'))
    plotter_acc = Plotter(phases=['pixacc', 'auc'], metrics=['accuracy'],
                          filename=os.path.join(opt.graph_path,f'acc_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.jpg'))
    
    # start training
    best_acc = 0
    best_epoch = 0
    model.train()
    
    try:
        for epoch in range(opt.epochs):
            index = epoch + 1 # easy for count
            # train
            loss = train_per_epoch(index, opt.epochs, model, train_generator, optimizer)
            scheduler.step(loss)
            plotter_loss.update(index,'train','loss',loss)
            
            # eval
            if index % opt.val_interval == 0:
                val_loss = val_per_epoch(index, model, val_generator)
                plotter_loss.update(index,'val','loss',val_loss)
            
            # test
            if index % opt.test_interval == 0:
                test_pixacc, test_auc = test_per_epoch(model, test_generator)
                plotter_acc.update(index,'pixacc', 'accuracy', test_pixacc)
                plotter_acc.update(index,'auc', 'accuracy', test_auc)
                
                if test_auc + opt.es_min_delta > best_acc:
                    best_acc = test_auc
                    best_epoch = index
            
            # early stop
            if index - best_epoch > opt.es_patience > 0:
                save_checkpoint(model, opt.log_path, f'{opt.prefix}_{index}.pth') 
                print('[Info] Stop training at epoch {}. The best auc achieved is {}'.format(index, best_acc))
                break
            
            plotter_loss.draw_curve()
            plotter_acc.draw_curve()
            # save 
            if index % opt.save_interval == 0 or index == opt.epochs:
                save_checkpoint(model, opt.log_path, f'{opt.prefix}_{index}.pth')

        print(f'Best AUC: {best_acc}, achieved at epoch {best_epoch}')

    except KeyboardInterrupt:
        save_checkpoint(model, opt.log_path, f'{opt.prefix}_{index}.pth') 

def train_per_epoch(epoch, epochs, model, generator, optimizer):
    epoch_loss = []
    progress_bar = tqdm(generator)
    num_iters = len(generator)
    for i, data in enumerate(progress_bar):
        try:
            rgb, nir, gt = data
            rgb = rgb.cuda()
            nir = nir.cuda()
            gt = gt.cuda()

            optimizer.zero_grad()
            output, loss = model(rgb, nir, gt)
            
            if loss == 0 or not torch.isfinite(loss):
                continue

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

            progress_bar.set_description(
                'Epoch: {}/{}. Iteration: {}/{}. Loss: {:.5f}'.format(
                    epoch, epochs, i + 1, num_iters, loss.item()))

        except Exception as e:
            print(e)
    return np.mean(epoch_loss)

def val_per_epoch(epoch, model, generator):   
    model.eval()
    val_loss = []
    for i, data in enumerate(generator):
        with torch.no_grad():
            rgb, nir, gt = data
            rgb = rgb.cuda()
            nir = nir.cuda()
            gt = gt.cuda()
            output, loss = model(rgb, nir, gt)
            
            if loss == 0 or not torch.isfinite(loss):
                continue
            val_loss.append(loss.item())
    
    val_loss = np.mean(val_loss)

    print('Val loss: {:.5f}'.format(val_loss))

    model.train()
    return val_loss

def test_per_epoch(model, generator):
    model.eval()
    test_pixacc = []
    test_auc = []
    for i, data in enumerate(generator):
        with torch.no_grad():
            rgb, nir, gt = data
            rgb = rgb.cuda()
            nir = nir.cuda()
            gt = gt.cuda()
            output, loss = model(rgb, nir, gt)
            
            pixacc = pix_acc(output, gt)
            auc = auc_roc(output, gt)
            
            if pixacc != 0:
                test_pixacc.append(pixacc)
            if auc != 0:
                test_auc.append(auc)
    
    test_pixacc = np.mean(test_pixacc)
    test_auc = np.mean(test_auc)
    
    print('Test pixel-wise accuracy: {:.5f}, AUC: {:.5f}'.format(test_pixacc, test_auc))
    model.train()
    return test_pixacc, test_auc

def save_checkpoint(model, path, name):
    torch.save(model.state_dict(), os.path.join(path, name))

def get_args():
    parser = argparse.ArgumentParser('Dual-branch Network for Multi-spectral SOD')
    parser.add_argument('--data_dir', type=str, default='/home/majin/datasets/MultiSpectralSOD/Images', help='data path')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=8, help='The number of images per batch among all devices')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--test_interval', type=int, default=5, help='Number of epoches between testing phases')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--log_path', type=str, default='./log/')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--graph_path', type=str, default='./graphs/')
    parser.add_argument('--prefix', type=str, default='base')
    parser.add_argument('--resume', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = get_args()
    torch.cuda.empty_cache()
    train(opt)
        