import cv2
from torchvision.transforms import ToTensor
from ultralytics import YOLO


model = YOLO("models/yolo/yolov8s-seg.pt")
img = cv2.imread("data/mlc_test_images/aadwyhrnrt.png")

pred = model.model(ToTensor()(img)[None, :])
a = 0
