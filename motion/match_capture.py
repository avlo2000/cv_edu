import cv2
import numpy as np


def process_and_match_orb(src, trg):
    feature_extractor = cv2.ORB_create(edgeThreshold=10)

    kp1, des1 = feature_extractor.detectAndCompute(src, None)
    kp2, des2 = feature_extractor.detectAndCompute(trg, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good, iso = find_homography(kp1, kp2, matches)
    match_picture = cv2.drawMatches(src, kp1, trg, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_picture, iso


def process_and_match_akaze(src, trg):
    feature_extractor = cv2.AKAZE_create()

    kp1, des1 = feature_extractor.detectAndCompute(src, None)
    kp2, des2 = feature_extractor.detectAndCompute(trg, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good, iso = find_homography(kp1, kp2, matches)
    match_picture = cv2.drawMatches(src, kp1, trg, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_picture, iso


def find_homography(kp1, kp2, matches):
    src_pts = []
    dst_pts = []
    for i, (m1, m2) in enumerate(matches):

        if m1.distance < 1.0 * m2.distance:
            src_pts.append(kp1[m1.queryIdx].pt)
            dst_pts.append(kp2[m1.trainIdx].pt)

    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
    transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    return [match[0] for match in matches], transformation_rigid_matrix


def select_roi(frame):
    roi_bb = cv2.selectROI("select_object", frame, showCrosshair=False)
    roi_bb = (roi_bb[0], roi_bb[1], roi_bb[0] + roi_bb[2], roi_bb[1] + roi_bb[3])
    roi = frame[roi_bb[1]:roi_bb[3], roi_bb[0]:roi_bb[2]]
    roi_shape = roi.shape
    return roi, roi_shape


def main():
    # cap = cv2.VideoCapture('../data/videos/toilet.mp4')
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    roi, roi_shape = select_roi(frame)
    pt_max = np.array([frame.shape[1], frame.shape[0]])
    pt_min = np.array([0, 0])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        match_res, affine = process_and_match_akaze(roi, frame)
        cv2.imshow("match_res", match_res)
        pt1 = (affine @ np.array([0, 0, 1])).astype(int)[:2]
        pt2 = [pt1[0] + roi_shape[1], pt1[1] + roi_shape[0]]
        pt1 = np.clip(pt1, pt_min, pt_max)
        pt2 = np.clip(pt2, pt_min, pt_max)
        # roi = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]

        frame_show = frame.copy()
        cv2.rectangle(frame_show, pt1=(pt1[0], pt1[1]), pt2=(pt2[0], pt2[1]), color=(255, 0, 0), thickness=4)
        cv2.imshow("detection_result", frame_show)

        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.waitKey(1) == ord(' '):
            while cv2.waitKey(1) != ord(' '):
                continue
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
