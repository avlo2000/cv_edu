import torch
from torch import nn


class HeightEstimator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_layer1 = nn.Sequential(nn.Conv2d(3, 16, (3, 3), stride=2), nn.ReLU6())
        self.img_layer2 = nn.Sequential(nn.Conv2d(16, 32, (3, 3), stride=2), nn.ReLU6())
        self.img_layer3 = nn.Sequential(nn.Conv2d(32, 64, (3, 3), stride=2), nn.ReLU6())
        self.img_layer4 = nn.Sequential(nn.Conv2d(64, 64, (3, 3), stride=2), nn.ReLU6())

        self.mask_layer1 = nn.Sequential(nn.Conv2d(1, 1, (3, 3), stride=2), nn.ReLU6())
        self.mask_layer2 = nn.Sequential(nn.Conv2d(1, 1, (3, 3), stride=2), nn.ReLU6())
        self.mask_layer3 = nn.Sequential(nn.Conv2d(1, 1, (3, 3), stride=2), nn.ReLU6())

        self.fcn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(61504, 64, bias=False),
            nn.ReLU6(),
            nn.Linear(64, 1, bias=False),
            nn.ReLU(),
        )

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.img_layer1(img + mask)
        x_m = self.mask_layer1(mask)

        x += x_m
        x = self.img_layer2(x)
        x_m = self.mask_layer2(x_m)

        x += x_m
        x = self.img_layer3(x)
        x_m = self.mask_layer3(x_m)

        x += x_m
        x = self.img_layer4(x)
        return self.fcn(x)


if __name__ == '__main__':
    m = HeightEstimator()
    m(torch.zeros([1, 3, 512, 512]), torch.zeros([1, 1, 512, 512]))
