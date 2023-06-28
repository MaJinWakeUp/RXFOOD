import os
import torchmetrics
import torch
from PIL import Image
import numpy as np
from torchvision import transforms as T

num_classes = 13

root_dir = '/home/majin/projects/CrossAttention/results/SOD'
model_name = 'DualBASNet'
gt_dir = os.path.join(root_dir, 'gt')
predict_dir = os.path.join(root_dir, model_name)
test_list = os.listdir(gt_dir)

max_f1 = []
mae = []

pr_curve = torchmetrics.classification.PrecisionRecallCurve(task='binary')

def compute_mae(pred, gt):
    pred = pred / 255
    gt = gt / 255
    return np.mean(np.abs(pred - gt))


for i in range(len(test_list)):
    gt = Image.open(os.path.join(gt_dir, test_list[i]))
    predict = Image.open(os.path.join(predict_dir, test_list[i]))
    
    gt = np.asarray(gt)
    predict = np.asarray(predict)

    mae.append(compute_mae(predict, gt))
    
    gt = T.ToTensor()(gt).unsqueeze(0)
    predict = T.ToTensor()(predict).unsqueeze(0)

    image_f1s = []
    for th in np.arange(0.01, 0.99, 0.02):
        f1 = torchmetrics.classification.BinaryF1Score(threshold=th)
        image_f1s.append(f1(predict, gt))
    max_f1.append(max(image_f1s))

    

print(f'MAE: {np.mean(mae)}')
print(f'Max F1: {np.mean(max_f1)}')






