#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:14:04 2022

@author: majin
"""
import torch
import torch.nn as nn
from model.BASNet import BASNet, BASNetHead, BASNetTail
from model.loss import SSIM, IOU
from model.FPN import FPNBackbone, FPNModule, FPNSemantic
from model.CPD_ResNet_models import CPD_ResNet, CPD_Head, CPD_Tail
from model.attentions import RXFOOD

class DualBASNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_channels_other=3):
        super().__init__()
        self.branch1 = BASNet(n_channels, n_classes)
        self.branch2 = BASNet(n_channels_other, n_classes)
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.ssim_loss = SSIM(window_size=11, size_average=True)
        self.iou_loss = IOU(size_average=True)
    
    def bce_ssim_loss(self, pred, target):
        bce_out = self.bce_loss(pred, target)
        ssim_out = 1-self.ssim_loss(pred, target)
        iou_out = self.iou_loss(pred, target)

        return bce_out + ssim_out + iou_out
    
    def loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, d7, gt):
        loss0 = self.bce_ssim_loss(d0, gt)
        loss1 = self.bce_ssim_loss(d1, gt)
        loss2 = self.bce_ssim_loss(d2, gt)
        loss3 = self.bce_ssim_loss(d3, gt)
        loss4 = self.bce_ssim_loss(d4, gt)
        loss5 = self.bce_ssim_loss(d5, gt)
        loss6 = self.bce_ssim_loss(d6, gt)
        loss7 = self.bce_ssim_loss(d7, gt)
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
        return loss0, loss

    def forward(self, x_rgb, x_nir,target):
        x_rgb0, x_rgb1, x_rgb2, x_rgb3, x_rgb4, x_rgb5, x_rgb6, x_rgb7 = self.branch1(x_rgb)
        x_nir0, x_nir1, x_nir2, x_nir3, x_nir4, x_nir5, x_nir6, x_nir7 = self.branch2(x_nir)
        output0 = (x_rgb0 + x_nir0) / 2
        output1 = (x_rgb1 + x_nir1) / 2
        output2 = (x_rgb2 + x_nir2) / 2
        output3 = (x_rgb3 + x_nir3) / 2
        output4 = (x_rgb4 + x_nir4) / 2
        output5 = (x_rgb5 + x_nir5) / 2
        output6 = (x_rgb6 + x_nir6) / 2
        output7 = (x_rgb7 + x_nir7) / 2
        loss0, loss = self.loss_fusion(output0, output1, output2, output3, output4, output5, output6, output7, target)
        loss0_rgb, loss_rgb = self.loss_fusion(x_rgb0, x_rgb1, x_rgb2, x_rgb3, x_rgb4, x_rgb5, x_rgb6, x_rgb7, target)
        loss0_nir, loss_nir = self.loss_fusion(x_nir0, x_nir1, x_nir2, x_nir3, x_nir4, x_nir5, x_nir6, x_nir7, target)
        loss0 = 0.8 * loss0 + 0.1 * loss0_rgb + 0.1 * loss0_nir
        loss = 0.8 * loss + 0.1 * loss_rgb + 0.1 * loss_nir

        return output0, [loss0, loss]
    
# import copy
class DualBASNetwithRXFOOD(nn.Module):
    def __init__(self, n_channels, n_classes, n_channels_other=3):
        super().__init__()
        self.branch1_head = BASNetHead(n_channels, n_classes)
        self.branch1_tail = BASNetTail(n_channels, n_classes)
        self.branch2_head = BASNetHead(n_channels_other, n_classes)
        self.branch2_tail = BASNetTail(n_channels_other, n_classes)
        self.att = RXFOOD(in_channels=[512,512,512], inter_channels=[128,128,128], num_scales=3, num_domains=2, attention='da')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.ssim_loss = SSIM(window_size=11, size_average=True)
        self.iou_loss = IOU(size_average=True)
    
    def bce_ssim_loss(self, pred, target):
        bce_out = self.bce_loss(pred, target)
        ssim_out = 1-self.ssim_loss(pred, target)
        iou_out = self.iou_loss(pred, target)

        return bce_out + ssim_out + iou_out
    
    def loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, d7, gt):
        loss0 = self.bce_ssim_loss(d0, gt)
        loss1 = self.bce_ssim_loss(d1, gt)
        loss2 = self.bce_ssim_loss(d2, gt)
        loss3 = self.bce_ssim_loss(d3, gt)
        loss4 = self.bce_ssim_loss(d4, gt)
        loss5 = self.bce_ssim_loss(d5, gt)
        loss6 = self.bce_ssim_loss(d6, gt)
        loss7 = self.bce_ssim_loss(d7, gt)
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
        # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f, l7: %3f" % 
        #         (loss0.data[0], loss1.data[0], loss2.data[0], loss3.data[0], loss4.data[0], loss5.data[0], loss6.data[0], loss7.data[0]))
        return loss0, loss

    def forward(self, x_rgb, x_nir,target):
        rgb_h1, rgb_h2, rgb_h3, rgb_h4, rgb_h5, rgb_h6 = self.branch1_head(x_rgb)
        nir_h1, nir_h2, nir_h3, nir_h4, nir_h5, nir_h6 = self.branch2_head(x_nir)
        
        rgb_out, nir_out = self.att([rgb_h4, rgb_h5, rgb_h6], [nir_h4, nir_h5, nir_h6])
        rgb_h4, rgb_h5, rgb_h6 = rgb_out
        nir_h4, nir_h5, nir_h6 = nir_out

        x_rgb0, x_rgb1, x_rgb2, x_rgb3, x_rgb4, x_rgb5, x_rgb6, x_rgb7 = self.branch1_tail(rgb_h1, rgb_h2, rgb_h3, rgb_h4, rgb_h5, rgb_h6)
        x_nir0, x_nir1, x_nir2, x_nir3, x_nir4, x_nir5, x_nir6, x_nir7 = self.branch2_tail(nir_h1, nir_h2, nir_h3, nir_h4, nir_h5, nir_h6)

        output0 = (x_rgb0 + x_nir0) / 2
        output1 = (x_rgb1 + x_nir1) / 2
        output2 = (x_rgb2 + x_nir2) / 2
        output3 = (x_rgb3 + x_nir3) / 2
        output4 = (x_rgb4 + x_nir4) / 2
        output5 = (x_rgb5 + x_nir5) / 2
        output6 = (x_rgb6 + x_nir6) / 2
        output7 = (x_rgb7 + x_nir7) / 2
        loss0, loss = self.loss_fusion(output0, output1, output2, output3, output4, output5, output6, output7, target)
        loss0_rgb, loss_rgb = self.loss_fusion(x_rgb0, x_rgb1, x_rgb2, x_rgb3, x_rgb4, x_rgb5, x_rgb6, x_rgb7, target)
        loss0_nir, loss_nir = self.loss_fusion(x_nir0, x_nir1, x_nir2, x_nir3, x_nir4, x_nir5, x_nir6, x_nir7, target)
        loss0 = 0.1 * loss0_rgb + 0.1 * loss0_nir + 0.8 * loss0
        loss = 0.1 * loss_rgb + 0.1 * loss_nir + 0.8 * loss

        return output0, [loss0, loss]


class DualFPN(nn.Module):
    def __init__(self, n_channels, n_classes, n_others=3, backbone='resnet101'):
        super().__init__()
        self.branch1_backbone = FPNBackbone(in_channels=n_channels, pretrained=True, back_bone=backbone)
        self.branch1_fpn = FPNModule(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
        self.branch1_semantic = FPNSemantic(in_channels=256, num_classes=n_classes)

        self.branch2_backbone = FPNBackbone(in_channels=n_others, pretrained=False, back_bone=backbone)
        self.branch2_fpn = FPNModule(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
        self.branch2_semantic = FPNSemantic(in_channels=256, num_classes=n_classes)

        self.loss_function = nn.BCEWithLogitsLoss(reduction='mean')

    def focal_loss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        logpt = -self.loss_function(logit, target)
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        return loss

    def forward(self, rgb, depth, target):
        rgb_c1, rgb_c2, rgb_c3, rgb_c4, rgb_c5 = self.branch1_backbone(rgb)
        depth_c1, depth_c2, depth_c3, depth_c4, depth_c5 = self.branch2_backbone(depth)
        # print("Backbone pass")

        rgb_p2, rgb_p3, rgb_p4, rgb_p5 = self.branch1_fpn(rgb_c2, rgb_c3, rgb_c4, rgb_c5)
        depth_p2, depth_p3, depth_p4, depth_p5 = self.branch2_fpn(depth_c2, depth_c3, depth_c4, depth_c5)
        # print("FPN pass")

        rgb_out = self.branch1_semantic(rgb_p2, rgb_p3, rgb_p4, rgb_p5)
        depth_out = self.branch2_semantic(depth_p2, depth_p3, depth_p4, depth_p5)
        output = (rgb_out + depth_out) / 2
        # print("Semantic pass")

        loss = 0.8 * self.loss_function(output, target) + 0.1 * self.loss_function(rgb_out, target) + 0.1 * self.loss_function(depth_out, target)
        return output, loss

class DualFPNwithRXFOOD(nn.Module):
    def __init__(self, n_channels, n_classes, n_others=3, backbone='resnet101'):
        super().__init__()
        self.branch1_backbone = FPNBackbone(in_channels=n_channels, pretrained=True, back_bone=backbone)
        self.branch1_fpn = FPNModule(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
        self.branch1_semantic = FPNSemantic(in_channels=256, num_classes=n_classes)

        self.branch2_backbone = FPNBackbone(in_channels=n_others, pretrained=False, back_bone=backbone)
        self.branch2_fpn = FPNModule(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
        self.branch2_semantic = FPNSemantic(in_channels=256, num_classes=n_classes)
        
        # add cross attention after backbone, before FPN
        # self.att = MSCDA2(in_channels=[256,512,1024,2048], inter_channels=[256,256,256,256], num_scales=4, num_domains=2, attention='da')

        # add cross attention after FPN, before semantic
        self.att = RXFOOD(in_channels=[256,256,256,256], inter_channels=[256,256,256,256], num_scales=4, num_domains=2, attention='da')
        # self.att = MSCDA3(in_channels=[256,256,256], in_spatials=[32,16,8], inter_channels=[256,256,256], num_domains=2, attention='da')
        
        self.loss_function = nn.BCEWithLogitsLoss(reduction='mean')

    
    def forward(self, rgb, depth, target):
        rgb_c1, rgb_c2, rgb_c3, rgb_c4, rgb_c5 = self.branch1_backbone(rgb)
        depth_c1, depth_c2, depth_c3, depth_c4, depth_c5 = self.branch2_backbone(depth)
        # print("Backbone pass")

        rgb_p2, rgb_p3, rgb_p4, rgb_p5 = self.branch1_fpn(rgb_c2, rgb_c3, rgb_c4, rgb_c5)
        depth_p2, depth_p3, depth_p4, depth_p5 = self.branch2_fpn(depth_c2, depth_c3, depth_c4, depth_c5)
        # print("FPN pass")

        rgb_out, depth_out = self.att([rgb_p2, rgb_p3, rgb_p4, rgb_p5], [depth_p2, depth_p3, depth_p4, depth_p5])
        rgb_p2, rgb_p3, rgb_p4, rgb_p5 = rgb_out
        depth_p2, depth_p3, depth_p4, depth_p5 = depth_out

        rgb_out = self.branch1_semantic(rgb_p2, rgb_p3, rgb_p4, rgb_p5)
        depth_out = self.branch2_semantic(depth_p2, depth_p3, depth_p4, depth_p5)
        output = (rgb_out + depth_out) / 2
        # print("Semantic pass")

        loss = 0.8 * self.loss_function(output, target) + 0.1 * self.loss_function(rgb_out, target) + 0.1 * self.loss_function(depth_out, target)
        return output, loss

class DualCPD(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.branch1 = CPD_ResNet(in_channels=in_channels)
        self.branch2 = CPD_ResNet(in_channels=in_channels)

        self.loss_function = nn.BCEWithLogitsLoss(reduction='mean')


    def forward(self, rgb, nir, target):
        rgb_m, rgb_out = self.branch1(rgb)
        nir_m, nir_out = self.branch2(nir)
        output = (rgb_out + nir_out) / 2
        # print("Semantic pass")

        loss = 0.5 * self.loss_function(output, target) + 0.25 * self.loss_function(rgb_out, target) + 0.25 * self.loss_function(nir_out, target)
        return output, loss


class DualCPDwithRXFOOD(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.branch1_head = CPD_Head(in_channels=in_channels)
        self.branch2_head = CPD_Head(in_channels=in_channels)
        self.branch1_tail = CPD_Tail()
        self.branch2_tail = CPD_Tail()

        self.att = RXFOOD(in_channels=[32,32,32], inter_channels=[32,32,32], num_scales=3, num_domains=2, attention='da')

        self.loss_function = nn.BCEWithLogitsLoss(reduction='mean')


    def forward(self, rgb, nir, target):
        rgb2_2, rgb3_2, rgb4_2 = self.branch1_head(rgb)
        nir2_2, nir3_2, nir4_2 = self.branch2_head(nir)
        
        rgb_out, nir_out = self.att([rgb2_2, rgb3_2, rgb4_2], [nir2_2, nir3_2, nir4_2])
        rgb2_2, rgb3_2, rgb4_2 = rgb_out
        nir2_2, nir3_2, nir4_2 = nir_out

        rgb_out = self.branch1_tail(rgb2_2, rgb3_2, rgb4_2)
        nir_out = self.branch2_tail(nir2_2, nir3_2, nir4_2)

        output = (rgb_out + nir_out) / 2

        # print("Semantic pass")
        loss = 0.8 * self.loss_function(output, target) + 0.1 * self.loss_function(rgb_out, target) + 0.1 * self.loss_function(nir_out, target)
        return output, loss