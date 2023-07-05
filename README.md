# RXFOOD
This is the pyTorch implementation for our paper [*"RXFOOD: Plug-in RGB-X Fusion for Object of Interest Detection"*](https://arxiv.org/abs/2306.12621).

We provide the training, testing, and evaluation code for RGB-NIR saliency object detection.

## Requirements
* pyTorch == 2.0.0
* numpy
* PILLOW
* [pysodmetrics](https://github.com/lartpang/PySODMetrics/tree/main)

## Dataset
Download the [RGBN-SOD dataset](https://tsllb.github.io/MultiSOD.html), and put dataset/train&val&test.txt under Image directory.

## Training
The default network for training is TBD<sub>CPD</sub>(Two Branch CPD Network), you can switch to different network by modifying line 54. You can modify parameters as shown in argparser.
```
python train_MSSOD.py --data_dir PATH_TO_DATASET
```

## Testing
Change the ckpt name, or download [the pretrained model](https://drive.google.com/file/d/1-aFeRggrFxHI2zcXnlIuN8_QU1ukR7Jr/view?usp=drive_link), and put it to `log/DualCPDwithRXFOOD_MSSOD.pth`.
```
python test_MSSOD.py
```

## Evaluation
```
python eval_api.py
```

## Citing RXFOOD
If you find this work helps you in your research, please use the following BibTex entry.
```BibTeX
@article{ma2023rxfood,
  title={RXFOOD: Plug-in RGB-X Fusion for Object of Interest Detection},
  author={Ma, Jin and Li, Jinlong and Guo, Qing and Zhang, Tianyun and Lin, Yuewei and Yu, Hongkai},
  journal={arXiv preprint arXiv:2306.12621},
  year={2023}
}
```