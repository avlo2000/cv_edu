from typing import Callable, Any

import cv2
import numpy as np


class ParamTrack:
    def __init__(self,
                 win_name: str,
                 param_name: str,
                 img: np.ndarray,
                 func: Callable[[Any, np.ndarray], np.ndarray],
                 parameter_space: np.ndarray):
        self.__img = img
        self.__func = func
        self.__parameter_space = parameter_space
        self.__win_name = f'{win_name} win'
        cv2.namedWindow(self.__win_name)
        cv2.createTrackbar(f'{param_name} track', self.__win_name, 0, len(parameter_space) - 1, self.__on_trackbar)
        self.__on_trackbar(0)

    def __on_trackbar(self, val):
        param = self.__parameter_space[int(val)]
        img = self.__func(param, self.__img)
        cv2.imshow(self.__win_name, img)


if __name__ == '__main__':
    test_img = cv2.imread('data/img5.jpg')

    def thres_call(thres, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresed = cv2.threshold(gray, thres, 50, cv2.THRESH_TOZERO_INV)
        return thresed

    def adaptive_thres_call(thres, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresed = cv2.adaptiveThreshold(gray, thres, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 1)
        return thresed

    def gauss_call(sigma, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blured = cv2.GaussianBlur(gray, (11, 11), sigma)
        return blured

    def bilateral_call(sigma, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blured = cv2.bilateralFilter(gray, 5, 10, sigma, cv2.BORDER_ISOLATED)
        return blured

    def canny_call(thres, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blured = cv2.Canny(gray, 1, thres)
        return blured

    def box_call(param, img):
        param = int(param // 2) * 2 + 1
        box = cv2.boxFilter(img, -1, ksize=(param, param))
        return box

    tracker = ParamTrack('Experiment', 'Index', test_img, canny_call, np.linspace(0, 255, 50))
    cv2.waitKey()
