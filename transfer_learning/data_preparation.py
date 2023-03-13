import torch.utils.data as data

import torchvision
from torchvision import transforms


def load_catsdogs_dataset(transform):
    train_data = torchvision.datasets.ImageFolder(root="../data/cats_dogs/train", transform=transform)
    test_data = torchvision.datasets.ImageFolder(root="../data/cats_dogs/test", transform=transform)
    val_data = torchvision.datasets.ImageFolder(root="../data/cats_dogs/val", transform=transform)
    return train_data, test_data, val_data


def load_mango_dataset(transform):
    train_data = torchvision.datasets.ImageFolder(root="../data/mangov1/train", transform=transform)
    test_data = torchvision.datasets.ImageFolder(root="../data/mangov1/test", transform=transform)
    val_data = torchvision.datasets.ImageFolder(root="../data/mangov1/valid", transform=transform)
    return train_data, test_data, val_data


def load_helmets_dataset(transform):
    train_data = torchvision.datasets.ImageFolder(root="../data/helmets/train", transform=transform)
    test_data = torchvision.datasets.ImageFolder(root="../data/helmets/test", transform=transform)
    val_data = torchvision.datasets.ImageFolder(root="../data/helmets/valid", transform=transform)
    return train_data, test_data, val_data


def load_caltech101(transform):
    dataset = torchvision.datasets.Caltech101(root="../data/caltech101", transform=transform, download=True)
    train_data, test_data, val_data = data.random_split(dataset, [0.8, 0.1, 0.1])
    return train_data, test_data, val_data


def load_dtd(transform):
    dataset = torchvision.datasets.DTD(root="../data/dtd", transform=transform, download=True)
    train_data, test_data, val_data = data.random_split(dataset, [0.8, 0.1, 0.1])
    return train_data, test_data, val_data
