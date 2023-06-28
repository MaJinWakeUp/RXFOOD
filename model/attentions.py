#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:07:42 2022

@author: majin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialHead(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            self.inter_channels = in_channels//2
        else:
            self.inter_channels = inter_channels
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
    
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(query, key) * torch.tensor(self.inter_channels ** -0.5)
        return energy

class SpatialTail(nn.Module):
    def __init__(self, in_channels, inter_channels, p=0.2):
        super().__init__()
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1) 
        self.dropout = nn.Dropout(p)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, energy, x):
        m_batchsize, C, height, width = x.size()
        attention = self.softmax(energy)
        # attention = energy
        # attention = 1 - attention
        attention = self.dropout(attention)
        value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        
        out = self.gamma*out + x
        out = out + x
        return out
    
class ChannelHead(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is not None:
            self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
    
    def forward(self, x):
        if self.inter_channels is not None:
            query = self.query_conv(x)
            key = self.key_conv(x)
        else:
            query = x
            key = x
        m_batchsize, C, height, width = query.size()
        query = query.view(m_batchsize, C, -1)
        key = key.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key) * torch.tensor(C ** -0.5)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # print(energy.size())
        return energy

class ChannelTail(nn.Module):
    def __init__(self, in_channel, inter_channel=None, p=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        if self.inter_channel is not None:
            self.value_conv = nn.Conv2d(in_channel, inter_channel, kernel_size=1)
            self.value_reconv = nn.Conv2d(inter_channel, in_channel, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    
    def forward(self, energy, x):
        if self.inter_channel is not None:
            value = self.value_conv(x)
        else:
            value = x
        m_batchsize, C, height, width = value.size()
        value = value.view(m_batchsize, C, -1)

        attention = self.softmax(energy)
        # attention = energy
        attention = self.dropout(attention)

        # print(attention.size())
        # print(value.size())
        
        out = torch.bmm(attention, value)
        out = out.view(m_batchsize, C, height, width)
        if self.inter_channel is not None:
            out = self.value_reconv(out)

        out = self.gamma*out + x
        out = out + x
        return out
    
class BasicAttention_partial(nn.Module):
    def __init__(self, in_channels, inter_channels=None, p=0.2, attention='sa'):
        super().__init__()
        if attention == 'sa':
            self.head = SpatialHead(in_channels, inter_channels)
            self.tail = SpatialTail(in_channels, inter_channels, p)
        elif attention == 'ca':
            self.head = ChannelHead(in_channels, inter_channels)
            self.tail = ChannelTail(in_channels, inter_channels, p)
        else:
            raise NotImplementedError
    
    def forward(self, x):
        energy = self.head(x)
        out = self.tail(energy, x)
        return out

class BasicAttention(nn.Module):
    def __init__(self, in_channels, inter_channels=None, p=0.2, attention='da'):
        super().__init__()
        self.atts = nn.ModuleList()
        if attention == 'da':
            self.atts.append(BasicAttention_partial(in_channels, inter_channels, attention='sa'))
            self.atts.append(BasicAttention_partial(in_channels, inter_channels, attention='ca'))
        elif attention == 'sa' or attention == 'ca':
            self.atts.append(BasicAttention_partial(in_channels, inter_channels, attention))
        else:
            raise NotImplementedError
    
    def forward(self, x):
        out = torch.zeros_like(x)
        for att in self.atts:
            out += att(x)
        return out

class RXFOOD_partial(nn.Module):
    def __init__(self, in_channels, inter_channels=None, num_scales=2, num_domains=2, attention='da', seperate=True):
        super().__init__()
        self.inter_channels = inter_channels
        
        self.attention = attention.lower()
        self.seperate = seperate
        if self.attention == 'sa':
            att_head = SpatialHead
            att_tail = SpatialTail
        elif self.attention == 'ca':
            att_head = ChannelHead
            att_tail = ChannelTail
        else:
            raise NotImplementedError
        self.num_scales = num_scales
        self.num_domains = num_domains
        self.heads = nn.ModuleList()
        self.tails = nn.ModuleList()
        for i in range(num_domains):
            domain_heads = nn.ModuleList()
            domain_tails = nn.ModuleList()
            for j in range(num_scales):
                domain_heads.append(att_head(in_channels[j], inter_channels[j]))
                domain_tails.append(att_tail(in_channels[j], inter_channels[j]))
            self.heads.append(domain_heads)
            self.tails.append(domain_tails)
        if self.seperate:
            self.energy_conv = nn.Conv2d(num_domains*num_scales, num_domains*num_scales, kernel_size=1)
        else:
            self.energy_conv = nn.Conv2d(num_domains*num_scales, 1, kernel_size=1)
    
    def forward(self, rgb_list, freq_list):
        if self.attention == 'sa':    
            scales = [x.size()[-1]**2 for x in rgb_list]
        elif self.attention == 'ca':
            if self.inter_channels[0] is None:
                scales = [x.size()[-1] for x in rgb_list]
            else:
                scales = [x for x in self.inter_channels]
        else:
            raise NotImplementedError
        if self.attention=='sa':
            ds = max(scales)
        elif self.attention=='ca':
            ds = max(scales)
        energy_list = []
        for i,x in enumerate(rgb_list):
            energy = self.heads[0][i](x).unsqueeze(1)
            energy = F.interpolate(energy, size=(ds, ds), mode='bilinear', align_corners=True)
            energy_list.append(energy)
            # print(energy.size())
        for i,x in enumerate(freq_list):
            energy = self.heads[1][i](x).unsqueeze(1)
            energy = F.interpolate(energy, size=(ds, ds), mode='bilinear', align_corners=True)
            energy_list.append(energy)
            
        enermap = torch.cat(energy_list, dim=1)
        enermap = self.energy_conv(enermap)
        
        if self.seperate:
            rgb_enermap = enermap[:,0:self.num_scales,:,:]
            freq_enermap = enermap[:,self.num_scales:,:,:]
        else:
            rgb_enermap = enermap.expand(-1, self.num_scales, -1, -1)
            freq_enermap = enermap.expand(-1, self.num_scales, -1, -1)
        
        rgb_ys = []
        freq_ys = []
        for i,x in enumerate(rgb_list):
            energy = rgb_enermap[:,i,:,:].unsqueeze(1)  
            energy = F.interpolate(energy, size=(scales[i], scales[i]), mode='bilinear', align_corners=True)
            energy = energy.squeeze(1)
            # print(energy.size())
            y = self.tails[0][i](energy, x)
            rgb_ys.append(y)
        for i,x in enumerate(freq_list):
            energy = freq_enermap[:,i,:,:].unsqueeze(1) 
            energy = F.interpolate(energy, size=(scales[i], scales[i]), mode='bilinear', align_corners=True)
            energy = energy.squeeze(1)
            # print(energy.size())
            y = self.tails[1][i](energy, x)
            freq_ys.append(y)
        return rgb_ys, freq_ys

class RXFOOD(nn.Module):
    def __init__(self, in_channels, inter_channels, num_scales=2, num_domains=2, attention='da', seperate=True):
        super().__init__()
        self.attention = attention
        self.atts = nn.ModuleList()
        self.seperate = seperate
        if attention == 'da':
            self.atts.append(RXFOOD_partial(in_channels, inter_channels, num_scales, num_domains, attention='sa', seperate=seperate))
            self.atts.append(RXFOOD_partial(in_channels, inter_channels, num_scales, num_domains, attention='ca', seperate=seperate))
        elif attention == 'sa' or attention == 'ca':
            self.atts.append(RXFOOD_partial(in_channels, inter_channels, num_scales, num_domains, attention))
        else:
            raise NotImplementedError

    def forward(self, rgb_list, freq_list):
        if self.attention == 'sa' or self.attention == 'ca':
            rgb_out, freq_out = self.atts[0](rgb_list, freq_list)
        elif self.attention == 'da':
            rgb_out = []
            freq_out = []
            rgb_out1, freq_out1 = self.atts[0](rgb_list, freq_list)
            rgb_out2, freq_out2 = self.atts[1](rgb_list, freq_list)
            for i in range(len(rgb_out1)):
                rgb_out.append((rgb_out1[i]+rgb_out2[i])/2)
                freq_out.append((freq_out1[i]+freq_out2[i])/2)
        return rgb_out, freq_out
