# GPF-YOLO
# Introduction
Here is the source code for an introduction to GPF-YOLO. 

## Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yolo python=3.6
```

- Then, activate the environment:
```Shell
conda activate yolo
```

- Requirements:
```Shell
pip install -r requirements.txt 
```

My environment:
- PyTorch = 1.9.1
- Torchvision = 0.10.1

At least, please make sure your torch is version 1.x.

## Experiments
### VOC

| Model        | Scale |   mAP/%  | Params/M |  FPS  |
|--------------|-------|----------|----------|-------|
| YOLOv1       |  640  |   63.4   |   37.8   |  45   |
| YOLOv2       |  640  |   79.8   |   53.9   |  40   |
| GPF-YOLO     |  640  |   80.6   |   43.9   |  70   |
| YOLOv3       |  640  |   82.0   |   167.4  |  78   |
| YOLOv4       |  640  |   83.6   |   162.7  |  66   |  
| YOLOX        |  640  |   84.6   |   155.4  |  69   |
| YOLOv7       |  640  |   85.5   |   144.6  |  56   |
