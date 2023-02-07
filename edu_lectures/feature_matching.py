import cv2
import numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 5
trg = cv2.imread('../data/ikea/images/room_scenes/ikea-a-bed-that-folds-away-to-be-a-sofa-by-day__1364309472525-s4.jpg')
templ = cv2.imread('../data/ikea/images/clock/img7.jpg')
trg = cv2.cvtColor(trg, cv2.COLOR_RGB2GRAY)
templ = cv2.cvtColor(templ, cv2.COLOR_RGB2GRAY)
templ = cv2.resize(templ, (templ.shape[0] // 10, templ.shape[0] // 10))

# Initiate ORB detector
feature_extractor = cv2.SIFT_create()
# find the keypoints and descriptors with ORB
kp1, des1 = feature_extractor.detectAndCompute(templ, None)
kp2, des2 = feature_extractor.detectAndCompute(trg, None)

# create BFMatcher object
bf = cv2.BFMatcher()
# Match descriptors.

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.9 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = templ.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    trg = cv2.polylines(trg, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)
img3 = cv2.drawMatches(templ, kp1, trg, kp2, good, None, **draw_params)
plt.imshow(img3, 'gray'), plt.show()
