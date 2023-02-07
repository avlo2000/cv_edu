import numpy as np
import cv2


def process(img):
    feature_extractor = cv2.FastFeatureDetector_create()
    # find the keypoints and descriptors with ORB
    kp = feature_extractor.detect(img, None)
    img = cv2.drawKeypoints(img, kp, None)
    return img


def process_and_match(img_prev, img):
    feature_extractor = cv2.ORB_create()
    img_prev = cv2.rotate(img_prev, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    kp1, des1 = feature_extractor.detectAndCompute(img_prev, None)
    kp2, des2 = feature_extractor.detectAndCompute(img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
    res = cv2.drawMatches(img_prev, kp1, img, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    res = cv2.rotate(res, cv2.ROTATE_90_COUNTERCLOCKWISE)
    res = cv2.resize(res, (1000, 500))
    return res


def process_and_match_akaze(img_prev, img):
    feature_extractor = cv2.AKAZE_create()

    kp1, des1 = feature_extractor.detectAndCompute(img_prev, None)
    kp2, des2 = feature_extractor.detectAndCompute(img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)
    res = cv2.drawMatches(img_prev, kp1, img, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return res


cap = cv2.VideoCapture('../data/videos/driving1.mp4')

ret, prev_frame = cap.read()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    draw = process_and_match_akaze(prev_frame, frame)
    cv2.imshow("stream", draw)

    prev_frame = frame
    if cv2.waitKey(20) == ord('q'):
        break
    if cv2.waitKey(20) == ord(' '):
        while cv2.waitKey(1) != ord(' '):
            continue
cap.release()
cv2.destroyAllWindows()
