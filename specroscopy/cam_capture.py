import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)

roi = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    view_img = frame

    if roi is None:
        roi = cv2.selectROI('select_roi', view_img)
        cv2.destroyWindow('select_roi')
        roi = np.array(roi)
        print(roi)
    cv2.rectangle(view_img, roi[:2], roi[:2] + roi[2:], (255, 0, 0), 3)
    cv2.imshow('view_img', view_img)
    cv2.waitKey(200)
