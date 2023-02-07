import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy import spatial


def plot_cv_img(input_image, output_image, plt_indexes):
    fig, ax = plt.subplots(nrows=plt_indexes[0], ncols=plt_indexes[1])
    ax[plt_indexes[0] - 1].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax[plt_indexes[0] - 1].set_title('L2 norm matching')
    ax[plt_indexes[0] - 1].axis('off')
    ax[plt_indexes[1] - 1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    ax[plt_indexes[1] - 1].set_title('Manhattan dist matching')
    ax[plt_indexes[1] - 1].axis('off')
    plt.show()


def calc_dogs(gray, octaves_cnt, gaussianKernelSize=5):
    prev = gray
    key_points = []
    des = []
    dogs = []
    for i in range(octaves_cnt):
        oct_blur = cv2.GaussianBlur(gray, (gaussianKernelSize, gaussianKernelSize), 0)
        diff = cv2.subtract(prev, oct_blur)
        prev = gray
        gray = oct_blur
        grad_polar = compute_gloh_grad(diff)
        dogs.append(diff)
        key_points, des = buid_gloh(grad_polar, key_points, des)

    return dogs


def compute_gloh_descr(gray, kp: cv2.KeyPoint):
    orient_hist_size = 16
    location_hist_size = 17
    win_sz = 4
    weight_multiplier = -0.5 / ((0.5 * win_sz) ** 2)

    sz_x = gray.shape[0]
    sz_y = gray.shape[1]

    x_l = int(-orient_hist_size / 2)
    x_r = int(orient_hist_size / 2)
    y_l = int(-orient_hist_size / 2)
    y_r = int(orient_hist_size / 2)
    step_ang = np.pi / 272

    angle = 360. - kp.angle
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    descriptor = [0.] * 272
    for x in range(x_l, x_r):
        for y in range(y_l, y_r):
            x_rot = int(kp.pt[0] + y * sin_angle + x * cos_angle)
            y_rot = int(kp.pt[1] + y * cos_angle - x * sin_angle)
            if 0 < x_rot < sz_x and 0 < y_rot < sz_y:
                x_left = max(x_rot - 1, 0)
                x_right = min(x_rot + 1, sz_x - 1)
                y_down = max(y_rot - 1, 0)
                y_up = min(y_rot + 1, sz_y - 1)

                diff_x = float(gray[x_left][y_rot]) - float(gray[x_right][y_rot])
                diff_y = float(gray[x_rot][y_down]) - float(gray[x_rot][y_up])
                vect = [0, 0]

                weight = np.exp(weight_multiplier * ((x_rot / location_hist_size) ** 2 + (y_rot / location_hist_size) ** 2))

                vect[0] = np.sqrt(
                    diff_x ** 2 + diff_y ** 2)
                if diff_x != 0:
                    vect[1] = np.arctan(diff_y / diff_x) + np.pi / 2
                else:
                    vect[1] = 0
                # - angle
                descriptor[int((vect[1]) / step_ang)] += vect[0]
    diff_sz = orient_hist_size * location_hist_size - len(descriptor)
    if diff_sz > 0:
        return None

    return descriptor


def gloh_detect_and_compute(gray):
    sift = cv2.SIFT_create()
    kps, descrs = sift.detectAndCompute(gray, None)
    d_orig = descrs
    descrs = [compute_gloh_descr(gray, kp) for kp in kps]
    kps_filtered = []
    descrs_filtered = []
    for i in range(len(kps)):
        if descrs[i] is not None:
            descrs_filtered.append(descrs[i])
            kps_filtered.append(kps[i])

    descrs = np.array(descrs_filtered)
    pca = PCA(n_components=128)
    pca.fit(descrs)
    descriptors_transformed = pca.transform(descrs)
    return kps_filtered, descriptors_transformed


def compute_gloh_grad(gray):
    xImgShape = gray.shape[0]
    yImgShape = gray.shape[1]

    grad_polar = np.ndarray(shape=(xImgShape, yImgShape, 2), dtype='float32')

    for x in range(xImgShape):
        xLeft = max(x - 1, 0)
        xRight = min(x + 1, xImgShape - 1)
        for y in range(yImgShape):
            yDown = max(y - 1, 0)
            yUp = min(y + 1, yImgShape - 1)
            diff_x = float(gray[xLeft][y]) - float(gray[xRight][y])
            diff_y = float(gray[x][yDown]) - float(gray[x][yUp])
            grad_polar[x][y][0] = np.sqrt(
                diff_x ** 2 + diff_y ** 2)
            if diff_x != 0:
                grad_polar[x][y][1] = np.arctan(diff_y / diff_x)
            else:
                grad_polar[x][y][1] = 0

    return grad_polar


def buid_gloh(grad_polar, key_points, descriptor):
    xImgShape = grad_polar.shape[0]
    yImgShape = grad_polar.shape[1]

    win_size = 4

    thres = 0.00005
    hist_size = 272
    step_ang = 360 / hist_size

    for x in range(0, xImgShape - win_size, win_size):
        for y in range(0, yImgShape - win_size, win_size):
            hist_angles = np.ndarray(shape=(hist_size,))
            hist_angles.fill(0.0)

            p = (x, y)
            ang = 0
            sz = 1
            for w_x in range(x, x + win_size):
                for w_y in range(y, y + win_size):
                    hist_angles[int(grad_polar[w_x][w_y][1] / step_ang)] += grad_polar[w_x][w_y][0]
                    if grad_polar[p[0]][p[1]][0] > grad_polar[w_x][w_y][0]:
                        p = (w_x, w_y)
                        ang = grad_polar[w_x][w_y][1]
                        sz = grad_polar[w_x][w_y][0]

            key_p = cv2.KeyPoint(p[0], p[1], sz, ang)
            descriptor.append(hist_angles)
            if sz < thres:
                key_points.append(key_p)
    return key_points, descriptor


def dist_cart(a, b):
    d = 0.0
    for ind in range(min(len(a), len(b))):
        d += (a[ind] - b[ind]) ** 2
    return np.sqrt(d)


def dist_manhattan(a, b):
    d = 0.0
    for ind in range(min(len(a), len(b))):
        d += np.abs(a[ind] - b[ind])
    return d


def match_fast(descrs1, descrs2, dist_delegate, num_take=20):
    matches = []
    tree = spatial.KDTree(descrs2)
    for i, k1 in enumerate(descrs1[:num_take]):
        res = tree.query(descrs1[i])
        matches.append(cv2.DMatch(_distance=dist_delegate(k1, descrs2[res[1]]), _imgIdx=0, _queryIdx=i, _trainIdx=res[1]))
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def match(descrs1, descrs2, dist_delegate, num_take=20):
    matches = []
    for i, k1 in enumerate(descrs1):
        for j, k2 in enumerate(descrs2):
            matches.append(cv2.DMatch(_distance=dist_delegate(k1, k2), _imgIdx=0, _queryIdx=i, _trainIdx=j))
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:num_take]


def main():
    img_paths = [('../data/Match' + str(i) + '1.jpg', '../data/Match' + str(i) + '2.jpg') for i in range(1, 4)]
    imgs = [(cv2.imread(p[0]), cv2.imread(p[1])) for p in img_paths]

    width = 400
    height = 400
    dsize = (width, height)
    for imgPair in imgs:
        img1 = imgPair[0]
        img2 = imgPair[1]
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1, des1 = gloh_detect_and_compute(gray1)
        kp2, des2 = gloh_detect_and_compute(gray2)

        # kp_img1 = img1
        # kp_img1 = cv2.drawKeypoints(img1, kp1, kp_img1)
        # kp_img2 = img2
        # kp_img2 = cv2.drawKeypoints(img2, kp2, kp_img2)

        matches_cart = match_fast(des1, des2, dist_cart, 30)
        matches_manh = match_fast(des1, des2, dist_manhattan, 30)

        cart_result = cv2.drawMatches(img1, kp1, img2, kp2, matches_cart, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        manh_result = cv2.drawMatches(img1, kp1, img2, kp2, matches_manh, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plot_cv_img(cart_result, manh_result, (1, 2))


if __name__ == '__main__':
    main()
