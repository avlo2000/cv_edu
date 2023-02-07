import cv2
from matplotlib import pyplot as plt

img2 = cv2.imread(
    '../data/ikea/images/room_scenes/ikea-a-bed-that-folds-away-to-be-a-sofa-by-day__1364309472525-s4.jpg')
img1 = cv2.imread('../data/ikea/images/clock/img7.jpg')
img1 = cv2.resize(img1, (img1.shape[0] // 8, img1.shape[0] // 8))
img2 = cv2.edgePreservingFilter(img2)
img1 = cv2.edgePreservingFilter(img1)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

plt.imshow(img3), plt.show()
