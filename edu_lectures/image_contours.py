import cv2
import matplotlib.pyplot as plt


def plot_with_hist(img):
    plt.subplot(1, 2, 1)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.show()


image = cv2.imread('../data/img5.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plot_with_hist(image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# img_filtered = cv2.GaussianBlur(image_gray, ksize=(5, 5), sigmaX=1, sigmaY=1)
# img_filtered = cv2.bilateralFilter(image_gray, 5, 10, 10, cv2.BORDER_REFLECT)
# img_filtered = cv2.edgePreservingFilter(image_gray)
img_filtered = image_gray

# plot_with_hist(img_filtered)

canny = cv2.Canny(img_filtered, 2, 55)
plot_with_hist(canny)

contours, _ = cv2.findContours(image=canny, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
cnt_thr = image.shape[0] * image.shape[1] / 15**2
filtered_contours = [cnt for cnt in contours if cv2.minAreaRect(cnt)[1][0] * cv2.minAreaRect(cnt)[1][1] > cnt_thr]

cv2.drawContours(image=image, contours=filtered_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.imshow("Detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

