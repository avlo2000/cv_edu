from torchvision import datasets
import torch
import matplotlib.pyplot as plt


axes = [
    (5, 5, 0),
    (5, 5, 1),
    (5, 5, 2),
    (16, 16, 0),
    (16, 16, 1),
    (16, 16, 2),
]


def main():
    dataset = datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
    )
    for axis in axes:
        data = torch.tensor(dataset.data, dtype=torch.float32) / 255.0
        bins, hist = torch.histogram(data[:, axis], bins=500)
        plt.plot(hist[:-1], bins)
        plt.title(repr(axis))
        plt.show()


if __name__ == '__main__':
    main()
