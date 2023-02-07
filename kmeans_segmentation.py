import cv2
import numpy as np
import distinctipy
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import dbscan


def prepare_flat_features(rgb):
    location_weight = 10
    x = np.linspace(0, location_weight, rgb.shape[1])
    y = np.linspace(0, location_weight, rgb.shape[0])
    location_x, location_y = np.meshgrid(x, y)
    location_x = np.expand_dims(location_x, axis=-1)
    location_y = np.expand_dims(location_y, axis=-1)
    img_with_locations = np.concatenate([rgb, location_x, location_y], axis=2)
    flatten = img_with_locations.reshape((-1, 5))
    flatten = np.float32(flatten)
    return flatten


def kmeans(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flatten = prepare_flat_features(rgb)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 6
    attempts = 10
    colors = distinctipy.get_colors(K)
    colors = np.array(colors)

    _, label, _ = cv2.kmeans(flatten, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    res = colors[label.flatten()]
    result_image = res.reshape((img.shape[0], img.shape[1], 3))
    return result_image


def segment_dbscan(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flatten = prepare_flat_features(rgb)

    _, label = dbscan(flatten, eps=5, min_samples=5)

    n_clusters = len(set(label)) + 1
    colors = distinctipy.get_colors(n_clusters)
    colors = np.array(colors)
    res = colors[label.flatten()]
    result_image = res.reshape((img.shape[0], img.shape[1], 3))
    return result_image


def neural_segmentation(img):
    net = cv2.dnn.readNetFromTensorflow("./models/inception/frozen_inference_graph.pb",
                                        "./models/inception/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
    width, height = 256, 256
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]
    print(detection_count)
    black_image = np.zeros_like(img)
    colors = distinctipy.get_colors(10)
    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = box[1]
        score = box[2]
        print(score)
        if score < 0.1:
            continue
        # Get box Coordinates
        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)
        roi = black_image[y: y2, x: x2]
        roi_height, roi_width, _ = roi.shape
        # Get the mask
        mask = masks[i, int(class_id)]
        mask = cv2.resize(mask, (roi_width, roi_height))
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)

        contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[int(class_id)]
        print(class_id)
        for cnt in contours:
            cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
    return img


def main():
    # path_to_sample = 'data/doors/doors1.jpg'
    # original = cv2.imread(path_to_sample)

    path_to_sample = 'data/cityscapes_data/train/16.jpg'

    label = cv2.imread(path_to_sample)[:, 256:]
    cv2.imshow("label", label)

    original = cv2.imread(path_to_sample)[:, :256]
    cv2.imshow("original", original)

    # dbscan_result = segment_dbscan(original)
    # cv2.imshow("dbscan_result", dbscan_result)

    kmeans_result = kmeans(original)
    cv2.imshow("kmeans_result", kmeans_result)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
