# RXFOOD
This is the pyTorch implementation for our paper [*"RXFOOD: Plug-in RGB-X Fusion for Object of Interest Detection"*](https://arxiv.org/abs/2306.12621).

We provide the training, testing, and evaluation code for RGB-NIR saliency object detection.

## Requirements
* pyTorch == 2.0.0
* numpy
* PILLOW
* [pysodmetrics](https://github.com/lartpang/PySODMetrics/tree/main)

## Training
The default network for training is TBD<sub>CPD</sub>(Two Branch CPD Network), you can switch to different network by modifying line 54.
```
python train_MSSOD.py --data_dir PATH_TO_DATASET
```

## Testing
Change the ckpt name, or download [the pretrained model](http://google.com unavailable now), and put it to `log/DualCPDwithRXFOOD.pth`.
```
python test_MSSOD.py
```

## Evaluation
```
python eval_api.py
```

