import numpy as np
import cv2


def kmeans(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    attempts = 10

    twoDimage = rgb.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)

    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return result_image


cap = cv2.VideoCapture('data/videos/traffic5.mp4')
detector = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=400)
# detector = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=1000)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame
    mask = detector.apply(roi)
    'mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 11)'
    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel1)
    cv2.imshow("thres", mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(cnt)
        if 50 < area < 200 and rect[1][0] < 60 and rect[1][1] < 60:
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    cv2.imshow("contours", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) == ord('q'):
        break
    if cv2.waitKey(1) == ord(' '):
        while cv2.waitKey(1) != ord(' '):
            continue
cap.release()
cv2.destroyAllWindows()
