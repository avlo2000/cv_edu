import sys

import torch
import torch.utils.data

import torchvision.transforms as transforms
import torch.nn as nn

from torch import optim
from torchvision import datasets

from torchsummary import summary

import distinctipy
import matplotlib.pyplot as plt

from autoencoders.cnn_autoencoder import CNNAutoEncoder
from autoencoders.autoencoder import Autoencoder
# from autoencoders.variational_autoencoder import Autoencoder

torch.use_deterministic_algorithms(True)
EPOCH_COUNT = 3
BATCH_SIZE = 64

data_transforms = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST(
    root='../data',
    train=True,
    transform=data_transforms,
    download=True,
)
test_data = datasets.MNIST(
    root='../data',
    train=False,
    transform=data_transforms
)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
in_shape: torch.Size = train_data[0][0].shape


model = Autoencoder(in_shape=in_shape, encoded_space_dim=2)
summary(model, input_size=in_shape, device='cpu')


def plot_latent_space():
    import pandas as pd
    import numpy as np

    x, y, labels = [], [], []
    for instance, label in test_data:
        latent = model.encoder(torch.unsqueeze(instance, dim=0)).detach().numpy()
        x.append(latent[:, 0])
        y.append(latent[:, 1])
        labels.append(label)

    df = pd.DataFrame({"x": np.squeeze(np.array(x)),
                       "y": np.squeeze(np.array(y)),
                       "label": np.squeeze(np.array(labels))})
    fig, ax = plt.subplots()

    classes = tuple(map(str, range(10)))
    for i, dff in df.groupby("label"):
        ax.scatter(dff['x'], dff['y'], s=50, label=classes[i])
    ax.legend()
    plt.show()


def train():
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(EPOCH_COUNT):
        print()
        for i, sample in enumerate(train_data_loader, 0):
            x, _ = sample
            x.requires_grad = True

            optimizer.zero_grad()

            x_pred = model(x)
            loss = loss_fn(x_pred, x)
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                sys.stdout.write(f'\r[Epoch: {epoch + 1}, Iter:{i + 1:5d}/{len(train_data_loader)}]'
                                 f' loss: {loss.item():.3f}'
                                 )
        checkpoint_path = f'./assets/checkpoint_{epoch + 1}.pt'
        print()
        print(f'Saving checkpoint to {checkpoint_path}')
        torch.save(model, checkpoint_path)
        print("-" * 100)

    print("-"*100)
    print("Testing...")


plot_latent_space()
train()
plot_latent_space()
