import matplotlib.pyplot as plt
import cv2


def sobel():
    cell = cv2.imread('../data/LowResDog.jpg')
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    gradx = cv2.Sobel(cell, cv2.CV_64F, 1, 0, ksize=1) # x axes derivative
    grady = cv2.Sobel(cell, cv2.CV_64F, 0, 1, ksize=1) # y axes derivative
    norm, angle = cv2.cartToPolar(gradx, grady, angleInDegrees=True)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cell, cmap='binary', origin='lower')

    plt.subplot(1, 2, 2)
    plt.imshow(norm, cmap='binary', origin='lower')

    plt.quiver(gradx, grady, color='blue')
    plt.savefig('gradient.png')
    plt.show()


if __name__ == '__main__':
    sobel()
