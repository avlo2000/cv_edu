import copy
from collections import OrderedDict

from torchvision import datasets
import torch
from torch import nn
from torch.utils import data
from torchvision.transforms import transforms

from learning_dynamics.utils import loss_drill
from transfer_learning import train_utils
import matplotlib.pyplot as plt
import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
BATCH_SIZE = 64
EPOCH_COUNT = 50


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
        nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(512, 10),
    )
    return model


def main():
    data_transforms = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root='../data',
        train=True,
        download=True,
        transform=data_transforms
    )
    train_data, test_data = data.random_split(dataset, [0.3, 0.7])
    train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    in_shape: torch.Size = dataset[0][0].shape

    trained_model = torch.load('assets/mnist_cnn.pkl').to(DEVICE)
    overfited_model = torch.load('assets/overfited_mnist_cnn.pkl').to(DEVICE)

    # overfited_model = create_model_cnn(in_shape).to(DEVICE)
    # overfited_model = train_utils.train(
    #                     model=trained_model,
    #                     epoch_count=EPOCH_COUNT,
    #                     train_data_loader=train_data_loader,
    #                     val_data_loader=test_data_loader,
    #                     test_data_loader=test_data_loader,
    #                     device=DEVICE
    #                 )
    # torch.save(trained_model, 'assets/overfited_mnist_cnn.pkl')

    # stupid_model = create_model_cnn(in_shape).to(DEVICE)

    weights, losses = loss_drill(trained_model, overfited_model, nn.CrossEntropyLoss(),
                                 test_data_loader, ticks_count=70, device=DEVICE)

    plt.title("Loss drill before train and after")
    plt.xlabel("ModelA to ModelB lerp weight")
    plt.ylabel("Total cross entropy loss")
    plt.plot(weights, losses, label=f"Loss drill num")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
