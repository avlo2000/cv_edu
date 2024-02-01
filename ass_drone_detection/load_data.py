from pathlib import Path
import re

import cv2
import numpy as np


def text_to_bbox(text: str) -> np.ndarray:
    return np.array([int(x) for x in text.split(', ')])


def load_labeled_video(path_to_vid: Path, path_to_ann: Path):
    cam = cv2.VideoCapture(str(path_to_vid))
    gt_text = open(path_to_ann)
    frames = []
    labels = []
    while True:
        ret, frame = cam.read()

        if not ret:
            break
        gt_line = gt_text.readline()
        pattern = r'\((.*?)\)'
        matches = re.findall(pattern, gt_line)
        bboxes = [text_to_bbox(m) for m in matches]
        frames.append(frame)
        labels.append(bboxes)
    return frames, labels


if __name__ == '__main__':
    def main():
        load_labeled_video(
            Path('../data/ass/Videos/Videos/Clip_30.mov'),
            Path('../data/ass/Video_Annotation/Video_Annotation/Clip_30_gt.txt')
        )
        a = 0


    main()
