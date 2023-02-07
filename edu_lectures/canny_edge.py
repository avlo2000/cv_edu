import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def canny_from_box(img, weak_th, strong_th):
    img = cv.GaussianBlur(img, (5, 5), 1.4)
    edges = cv.Canny(img, weak_th, strong_th)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def canny(img, weak_th, strong_th):
    img = cv.GaussianBlur(img, (5, 5), 1.4)

    gx = cv.Sobel(np.float32(img), cv.CV_64F, 1, 0, 3)
    gy = cv.Sobel(np.float32(img), cv.CV_64F, 0, 1, 3)

    mag, ang = cv.cartToPolar(gx, gy, angleInDegrees=True)

    height, width = img.shape

    edges = mag
    for i_x in range(width):
        for i_y in range(height):

            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            neigh1_x, neigh1_y = -1, -1
            neigh2_x, neigh2_y = -1, -1
            if grad_ang <= 22.5:
                neigh1_x, neigh1_y = i_x - 1, i_y
                neigh2_x, neigh2_y = i_x + 1, i_y
            elif 22.5 < grad_ang <= (22.5 + 45):
                neigh1_x, neigh1_y = i_x - 1, i_y - 1
                neigh2_x, neigh2_y = i_x + 1, i_y + 1
            elif (22.5 + 45) < grad_ang <= (22.5 + 90):
                neigh1_x, neigh1_y = i_x, i_y - 1
                neigh2_x, neigh2_y = i_x, i_y + 1
            elif (22.5 + 90) < grad_ang <= (22.5 + 135):
                neigh1_x, neigh1_y = i_x - 1, i_y + 1
                neigh2_x, neigh2_y = i_x + 1, i_y - 1
            elif (22.5 + 135) < grad_ang <= (22.5 + 180):
                neigh1_x, neigh1_y = i_x - 1, i_y
                neigh2_x, neigh2_y = i_x + 1, i_y

            if width > neigh1_x >= 0 and height > neigh1_y >= 0:
                if mag[i_y, i_x] < mag[neigh1_y, neigh1_x]:
                    edges[i_y, i_x] = 0
                    continue

            if width > neigh2_x >= 0 and height > neigh2_y >= 0:
                if mag[i_y, i_x] < mag[neigh2_y, neigh2_x]:
                    edges[i_y, i_x] = 0

    edges = 255. * edges / edges.max()

    edges = cv.threshold(edges, weak_th, strong_th, type=cv.THRESH_BINARY)[1] * 255.
    edges = edges.astype(np.uint8)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    image = cv.imread('../data/img4.jpg', 0)
    canny_from_box(image, 50, 200)
