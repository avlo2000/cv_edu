from ultralytics import YOLO


model = YOLO(r"C:\Users\PC\Development\sc_ml\runs\segment\train24\weights\best.pt")
if __name__ == '__main__':
    results = model.train(data="data/YOLODataset/dataset.yaml", epochs=20, imgsz=512, workers=0, verbose=True, plots=True)
