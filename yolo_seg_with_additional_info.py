import numpy as np
from ultralytics import YOLO
from torchvision.transforms import ToTensor

model = YOLO("yolov8n_seg_contest.yaml", task='segment')
# model = YOLO("yolov8n-seg.pt")
img = np.zeros([512, 512, 3], dtype=np.uint8)
img_t = ToTensor()(img)[None, :]
res = model.model(img_t)
a = 0
