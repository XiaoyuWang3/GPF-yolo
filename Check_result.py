import torch as t
#import torch_directml
import numpy
#dml = torch_directml.device()

checkpoints = t.load('weights/voc/yolov7/yolov7_best.pth')
print(checkpoints['mAP'],checkpoints['epoch'])