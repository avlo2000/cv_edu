import glob

import cv2
import albumentations as alb
import uuid


AUG_MUL = 1000

augment = alb.Compose(
    [
        alb.VerticalFlip(p=0.5),
        alb.RandomCropFromBorders(crop_left=0.05, crop_right=0.05, crop_bottom=0.05, crop_top=0.05),
        alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.08, rotate_limit=0, p=0.5),
        alb.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        alb.RandomBrightnessContrast(p=0.5),
    ]
)

preproc = alb.Compose(
    [
        alb.Resize(64, 64),
        alb.ToGray(p=1)
    ]
)

positives = [cv2.imread(path) for path in glob.glob("data/pos/*.jpg")]
negatives = [cv2.imread(path) for path in glob.glob("data/neg/*.jpg")]


for i in range(AUG_MUL):
    for img in positives:
        augmented = augment(image=img)['image']
        prep = preproc(image=img)['image']
        cv2.imwrite(f'data_augmented_x1000/pos/{uuid.uuid4()}.jpg', prep)

for i in range(AUG_MUL):
    for img in negatives:
        augmented = augment(image=img)['image']
        prep = preproc(image=img)['image']
        cv2.imwrite(f'data_augmented_x1000/neg/{uuid.uuid4()}.jpg', prep)
