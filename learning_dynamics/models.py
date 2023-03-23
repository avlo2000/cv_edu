from torch import nn


def create_model_bigger_cnn(in_shape):
    model = nn.Sequential(
        nn.Conv2d(in_shape[0], 8, kernel_size=3, padding=1, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(128, 10),
    )
    return model


def create_model(in_shape):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_shape.numel(), 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 10),
    )
    return model


def create_model_cnn(in_shape):
    model = nn.Sequential(
        nn.Conv2d(in_shape[0], 8, kernel_size=3, padding=1, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(512, 10),
    )
    return model
