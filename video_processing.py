import numpy as np
import cv2


def exclude_isolated(img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 10

    img2 = np.zeros(output.shape)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2


cap = cv2.VideoCapture('data/videos/traffic5.mp4')

canny_thres_low = 250
canny_thres_high = 255

ret, prev_frame = cap.read()
gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_canny = cv2.Canny(gray, canny_thres_low, canny_thres_high)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, canny_thres_low, canny_thres_high)

    cv2.imshow('canny', canny)
    diff = cv2.absdiff(canny, prev_canny)
    diff = exclude_isolated(diff)
    cv2.imshow('diff', diff)

    prev_canny = canny
    if cv2.waitKey(1) == ord('q'):
        break
    if cv2.waitKey(1) == ord(' '):
        while cv2.waitKey(1) != ord(' '):
            continue
cap.release()
cv2.destroyAllWindows()
