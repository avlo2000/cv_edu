import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, fftpack

KERNEL_SIZE = 13

print(np.get_include())


def plot_cv_img(input_image, output_image, plt_indexes):
    """
    Converts an image from BGR to RGB and plots
    """
    fig, ax = plt.subplots(nrows=plt_indexes[0], ncols=plt_indexes[1])
    ax[plt_indexes[0] - 1].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax[plt_indexes[0] - 1].set_title('Input Image')
    ax[plt_indexes[0] - 1].axis('off')
    ax[plt_indexes[1] - 1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    ax[plt_indexes[1] - 1].set_title('Box Filter (' + str(KERNEL_SIZE) + ' by ' + str(KERNEL_SIZE) + ')')
    ax[plt_indexes[1] - 1].axis('off')
    plt.show()


def convolute(image, kernel: np.array):
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    assert xKernShape % 2 == 1 and yKernShape % 2 == 1
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    xOutput = xImgShape - xKernShape
    yOutput = yImgShape - yKernShape

    midX = int(xKernShape / 2)
    midY = int(yKernShape / 2)

    channels = 1
    if len(image.shape) == 3:
        channels = image.shape[2]
    output = np.zeros(shape=(xOutput, yOutput, channels), dtype='uint8')

    def calc_val(x, y, chan=3):
        s = np.zeros(shape=(chan,), dtype='uint8')
        for iK in range(x - midX, x + midX + 1):
            for jK in range(y - midY, y + midY + 1):
                for c in range(chan):
                    s[c] += kernel[iK - x + midX][jK - y + midY] * image[iK][jK][c]
        return s

    for i in range(midX, xImgShape - midX):
        for j in range(midY, yImgShape - midY):
            output[i - midX - 1][j - midY - 1] = calc_val(i, j, channels)
    ### O(nk * mk * nt * mt)
    return output


def entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = np.e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def fft_convolute(image, kernel: np.array):
    channels = 1
    if len(image.shape) == 3:
        channels = image.shape[2]

    sz = (image.shape[0] - kernel.shape[0], image.shape[1] - kernel.shape[1])

    new_kernel = np.ndarray(shape=(kernel.shape[0], kernel.shape[1], channels))
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            new_kernel[i][j].fill(kernel[i][j])

    kernel = new_kernel
    kernel = np.pad(kernel, (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)), 'constant')
    kernel = fftpack.ifftshift(kernel)

    result = np.real(fftpack.ifft2(fftpack.fft2(image) * fftpack.fft2(kernel)))
    return result


def main():
    img_paths = ['../data/img' + str(i) + '.jpg' for i in range(1, 7)]
    imgs = [cv2.imread(p) for p in img_paths]

    shape = (KERNEL_SIZE, KERNEL_SIZE)
    kernel = np.ndarray(shape=shape)
    kernel.fill(1 / KERNEL_SIZE ** 2)
    if KERNEL_SIZE % 2 == 0:
        shape = (KERNEL_SIZE + 1, KERNEL_SIZE + 1)
        kernel = np.ndarray(shape=shape)
        kernel.fill(0)
        for i in range(KERNEL_SIZE):
            for j in range(KERNEL_SIZE):
                kernel[i][j] = 1 / KERNEL_SIZE ** 2

    width = 400
    height = 400
    dsize = (width, height)

    for img in imgs:
        img = cv2.resize(img, dsize)

        print("ENTROPY ORIGINAL: " + str(entropy(img, 2)))
        blur1 = cv2.GaussianBlur(img, (KERNEL_SIZE, KERNEL_SIZE), sigmaX=100.0)
        # blur2 = convolute(img, kernel)

        print("ENTROPY CONVOLUTED: " + str(entropy(blur1, 2)))
        print("----------------")

        plot_cv_img(img, blur1, (1, 2))
        # plot_cv_img(img, blur2, (1, 2))


if __name__ == '__main__':
    main()
