import cv2
import numpy as np


IMSHOW = False


def detect_crosses(original) -> [np.ndarray]:
    img = original.copy()
    red = img[:, :, 2]
    red[red == 0] = 255
    red[red != 255] = 0
    red = red.astype(float) / 255

    kernel = np.zeros([7, 7], dtype=float)
    np.fill_diagonal(kernel, 1)
    np.fill_diagonal(np.fliplr(kernel), 1)

    crosses = cv2.filter2D(red, ddepth=-1, kernel=kernel)
    # if IMSHOW:
    #     cv2.imshow("crosses", crosses)
    crosses = (crosses == 0).astype('uint8')
    contours, _ = cv2.findContours(crosses, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect_conts(original) -> [np.ndarray]:
    img_gaussian = original.copy()
    img_gaussian = cv2.cvtColor(img_gaussian, cv2.COLOR_RGB2GRAY)
    img_gaussian[img_gaussian != 0] = 255
    img_gaussian = 255 - img_gaussian

    img_gaussian = cv2.GaussianBlur(img_gaussian, (5, 5), sigmaX=1, sigmaY=1)
    kernel = np.ones([5, 5], dtype=img_gaussian.dtype)
    thick = cv2.filter2D(img_gaussian, ddepth=-1, kernel=kernel)

    canny = cv2.Canny(thick, 30, 255)
    # if IMSHOW:
    #     cv2.imshow("canny", canny)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def main():
    import glob
    with open('true_labels.txt', 'w') as labels_file:
        for img_path in glob.glob('../data/somatic_edu/hidden/*.png'):
            original = cv2.imread(img_path)
            crosses = detect_crosses(original)
            conts = detect_conts(original)
            labels_file.write(img_path.split('\\')[-1] + f" {len(crosses)} {len(conts)}\n")
            if IMSHOW:
                cv2.drawContours(original, conts, -1, (0, 255, 0), 3)
                cv2.drawContours(original, crosses, -1, (0, 255, 0), 3)
                cv2.imshow("contours", original)
                cv2.waitKey(0)


if __name__ == '__main__':
    main()
