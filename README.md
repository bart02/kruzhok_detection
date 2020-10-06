# Kruzhok detection with Faster R-CNN + SVM HOG Classification
KD logo detection for KRUZHOK.PRO selection
Team: Mudreci

## Abstract
We use a 2-level model for detecting potential logos + classification for increasing quality.  
Dataset for detector was compiled using augmented images from social networks (we labeled them using LabelImg) and open datasets with logos.  
Dataset for svm-classifier was generated from detector's predictions on previous dataset + open datasets for balancing classes.  
We used transfer learning for R-CNN (ResNet-50).  
Dataset: https://drive.google.com/drive/folders/1AkkbzrETito3GuVp52L6hyAzLv_tiHP2?usp=sharing

## Usage

```
usage: run.py [-h] [-d DIRPATH] [-f FILEPATH]

optional arguments:
  -h, --help   show this help message and exit
  -d DIRPATH   path to directory with images
  -f FILEPATH  path to image
```


## Requirements
For cpu:
`pip install -r cpu_requirements.txt`

For gpu:
1) Install latest gpu drivers + CUDA for torch
2) `pip install -r gpu_requirements.txt`
