import glob

import cv2
import numpy as np
import albumentations as alb
import torch
from torchvision.transforms import ToTensor


def main():
    samples = [(cv2.imread(path), path) for path in glob.glob("hidden/*.jpg")]

    preproc = alb.Compose(
        [
            alb.Resize(64, 64),
            alb.ToGray(p=1)
        ]
    )
    model = torch.load('assets/big_bro.pt')
    model.eval()

    to_tensor = ToTensor()
    for img, path in samples:
        x = to_tensor(preproc(image=img)['image'])
        x = torch.unsqueeze(x, dim=0)
        pred = model(x).item()
        print(f"{path} {pred > 0.5}")


if __name__ == '__main__':
    main()
