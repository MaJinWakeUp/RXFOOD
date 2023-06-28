#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:15:00 2021

@author: majin
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def pix_acc(pred, mask):
    all_pixacc = []
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
        mask = mask.cpu().numpy()
    
    pred = pred.squeeze()
    mask = mask.squeeze().astype(np.int64)
    
    thres = np.arange(0,1,0.05)
    num_thres = len(thres)
    for i in range(num_thres):
        batch_pixacc = 0
        pred_mask = pred>thres[i]
        
        num, width, height = mask.shape
        for i in range(num):
            y_pred = pred_mask[i,:,:].reshape(-1)
            y_true = mask[i,:,:].reshape(-1)
            # print(y_true)
            # print(y_pred)
            img_pixacc = accuracy_score(y_true, y_pred)
            batch_pixacc += img_pixacc
        
        batch_pixacc = batch_pixacc/num
        all_pixacc.append(batch_pixacc)
    return np.max(all_pixacc)

def auc_roc(pred, mask):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
        mask = mask.cpu().numpy()
    pred = pred.squeeze()
    mask = mask.squeeze().astype(np.int64)
    batch_auc = 0
    
    num, width, height = mask.shape
    count = num
    for i in range(num):
        y_score = pred[i,:,:].reshape(-1)
        y_true = mask[i,:,:].reshape(-1)
        # if np.sum(y_true)==0 or np.sum(y_true)==380*380:
        #     count -= 1
        #     continue
        batch_auc += roc_auc_score(y_true, y_score)
        
    batch_auc = batch_auc/count
    return batch_auc

def auc_cls(pred, label):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
    pred = pred.squeeze()
    label = label.squeeze().astype(np.int64)
    
    batch_auc = roc_auc_score(label.reshape(-1), pred.reshape(-1))
    
    return batch_auc

def seg_f1(y_true, y_pred):
    TP = 0; FP = 0;TN = 0; FN = 0

    for i in range(len(y_true)): 
        if y_true[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
           FP += 1
        if y_true[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
           FN += 1
    
    f1 = (2*TP) / float(2*TP + FN + FP)
    return f1

def f1_score(pred, mask, cindex=1, replace=1):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
        mask = mask.cpu().numpy()
    # batch_f1 = 0
    pred = pred.squeeze()
    mask = mask.squeeze().astype(np.int64)
    num, width, height = mask.shape
    
    thres = np.arange(0,1,0.05)
    num_thres = len(thres)
    batch_f1 = np.zeros([num_thres,])
    for i in range(num):
        for j in range(num_thres):
            pred_mask = pred[i,:,:]>thres[j]
            y_pred = pred_mask.astype(np.int32).reshape(-1)
            y_true = mask[i,:,:].reshape(-1)
            f1 = seg_f1(y_true, y_pred)
            batch_f1[j] += f1

    batch_f1 = batch_f1/num
    return batch_f1
            
def f1_cls(pred, label, replace=1):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
    # batch_f1 = 0
    pred = pred.squeeze()
    label = label.squeeze().astype(np.int64)
    
    thres = np.arange(0,1,0.05)
    num_thres = len(thres)
    batch_f1 = np.zeros([num_thres,])

    for j in range(num_thres):
        pred_label = pred>thres[j]
        y_pred = pred_label.astype(np.int32).reshape(-1)
        y_true = label.reshape(-1)
        f1 = seg_f1(y_true, y_pred)
        batch_f1[j] = f1

    batch_f1 = np.max(batch_f1)
    return batch_f1
            