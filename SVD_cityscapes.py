import numpy as np
import tqdm

import cv2 as cv
import glob

import matplotlib.pyplot as plt

img = cv.imread('data/helmets/train/Helmet/00799_jpg.rf.a1553100105d6ae289921a753924350c.jpg')

rgb = cv.split(img)
rate = 0.1

for sub_img in rgb:
    U, S, VT = np.linalg.svd(sub_img, full_matrices=False)
    S = np.diag(S)
    r = int(rate * S.shape[0])
    res = U[:, :r] @ S[:r, :r] @ VT[:r, :]

    compression_rate = U[:, :r].size + S[:r, :r].size + VT[:r, :].size
    print(img.size / compression_rate)

    plt.subplot(1, 2, 1)
    plt.imshow(res)
    plt.subplot(1, 2, 2)
    plt.imshow(sub_img)
    plt.show()
