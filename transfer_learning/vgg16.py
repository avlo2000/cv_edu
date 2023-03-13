import torch
from torch import nn


class VGG16ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            VGG16ConvBlock(3, 64, kernel_size=3, stride=1, padding=1),
            VGG16ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG16ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            VGG16ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG16ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            VGG16ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            VGG16ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG16ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            VGG16ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            VGG16ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG16ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            VGG16ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            VGG16ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7 * 7 * 512, 1024),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(1024, num_classes))

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return torch.sigmoid(out)
