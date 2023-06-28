import torch
import torch.nn as nn
import torch.nn.functional as F

class BasePUP(nn.Module):
    def __init__(self, inter_channels=[1792,448,160,56,32,24], inter_size=[12,24,48,95,190,380], num_classes=1, dropout=0.2):
        super().__init__()
        self.decoders = nn.ModuleList()
        self.sizes = inter_size
        self.preds = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.decoders.append(nn.Sequential(
                                    nn.Conv2d(inter_channels[0], inter_channels[1], 1),
                                    nn.BatchNorm2d(inter_channels[1]),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(size=(inter_size[0], inter_size[0]), mode="bilinear", align_corners=True)
                                ))
        self.preds.append(nn.Conv2d(inter_channels[1], num_classes, 1))
        for i in range(1,len(inter_channels)):
            out_channel = inter_channels[i+1] if i<len(inter_channels)-1 else inter_channels[-1]
            self.decoders.append(nn.Sequential(
                                    nn.Conv2d(inter_channels[i]*2, out_channel, 1),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(size=(inter_size[i], inter_size[i]), mode="bilinear", align_corners=True)
                                ))
            self.preds.append(nn.Conv2d(out_channel, num_classes, 1))

    def forward(self, x):
        output = dict()
        scales = len(self.decoders)
        temp = self.decoders[0](self.dropout(x['reduction_'+str(scales)]))
        temp_pred = self.preds[0](temp)
        output['reduction_'+str(scales)] = temp_pred
        for i in range(1,scales):
            key = 'reduction_'+str(scales-i)
            temp = torch.cat([temp, x[key]], dim=1)
            temp = self.decoders[i](self.dropout(temp)) 
            temp_pred = self.preds[i](temp)
            output[key] = temp_pred
        return output

class MultPUP(nn.Module):
    def __init__(self, inter_channels=[1792,448,160,56,32,24], inter_size=[12,24,48,95,190,380], num_classes=1, dropout=0.2):
        super().__init__()
        self.decoders = nn.ModuleList()
        self.sizes = inter_size
        self.dropout = nn.Dropout(p=dropout)
        for i in range(len(inter_channels)):
            self.decoders.append(nn.Sequential(
                nn.Conv2d(inter_channels[i], inter_channels[i]//2, 1),
                nn.BatchNorm2d(inter_channels[i]//2),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(inter_size[i], inter_size[i]), mode="bilinear", align_corners=True),
                nn.Conv2d(inter_channels[i]//2, num_classes, 1),
                nn.Sigmoid()
                ))
        
    
    def forward(self, x):
        output = dict()
        temp = self.decoders[0](self.dropout(x['reduction_6']))
        output['reduction_6'] = temp
        
        for i in range(1,len(self.decoders)):
            key = 'reduction_'+str(6-i)
            temp = temp * x[key]
            temp = self.decoders[i](temp)
            output[key] = temp
        return output
    


class ResnetDecoderPUP(nn.Module):
    def __init__(self, inter_channels=256, num_classes=1):
        super().__init__()
        self.channels = min(128, inter_channels//2)
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(inter_channels, self.channels, 3, padding=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, 3, padding=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_3 = nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, 3, padding=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, 3, padding=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_5 = nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, 3, padding=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.channels, num_classes, 1),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.decoder_5(x)
        return x

class FPNDecoderPUP(nn.Module):
    def __init__(self, inter_channels=256, num_classes=1):
        super().__init__()
        self.channels = min(128, inter_channels // 2)
        self.conv_p5 = nn.Conv2d(inter_channels, self.channels, 1, padding=0)
        self.decoder_p5 = nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, 3, padding=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                    )
        self.conv_p4 = nn.Conv2d(inter_channels, self.channels, 1, padding=0)
        self.decoder_p4 = nn.Sequential(
                    nn.Conv2d(self.channels*2, self.channels, 3, padding=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                    )
        self.conv_p3 = nn.Conv2d(inter_channels, self.channels, 1, padding=0)
        self.decoder_p3 = nn.Sequential(
                    nn.Conv2d(self.channels*2, self.channels, 3, padding=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                    )
        # self.conv_p2 = nn.Conv2d(inter_channels, self.channels, 1, padding=0)
        self.decoder_p2 = nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, 3, padding=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                    )
        self.decoder_final = nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, 3, padding=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(self.channels, num_classes, 1),
                    )

    def forward(self, low_xs):
        p3, p4, p5 = low_xs
        p5 = self.conv_p5(p5)
        x = self.decoder_p5(p5)
        
        p4 = self.conv_p4(p4)
        x = torch.cat([x, p4], dim=1)
        x = self.decoder_p4(x)
        
        p3 = self.conv_p3(p3)
        x = torch.cat([x, p3], dim=1)
        x = self.decoder_p3(x)
        
        # p2 = self.conv_p2(p2)
        # x = torch.cat([x, p2], dim=1)
        x = self.decoder_p2(x)
        
        x = self.decoder_final(x)
        return x
    
class FPNDecoderCLS(nn.Module):
    def __init__(self, inter_channels=256, num_classes=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear1 = nn.Linear(inter_channels*3, inter_channels)
        self.linear2 = nn.Linear(inter_channels, num_classes)
        
    def forward(self, low_xs):
        p3, p4, p5 = low_xs
        p3 = self.pool(p3)
        p4 = self.pool(p4)
        p5 = self.pool(p5)
        x = torch.cat([p3,p4,p5], dim=1)
        x = torch.flatten(x, start_dim=1)
        
        x = self.linear2(F.relu(self.linear1(x)))
        return x

class ClsDecoder(nn.Module):
    def __init__(self, in_channels=1792, num_classes=1, dropout=0.2):
        super().__init__()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(dropout)
        self._fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x