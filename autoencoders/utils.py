import sys

import numpy as np
import torch

from matplotlib import pyplot as plt
from torch import nn, optim


def plot_latent_space(model, data, device, classes):
    import pandas as pd

    x, y, labels = [], [], []
    for instance, label in data:
        latent = model.encoder(torch.unsqueeze(instance, dim=0).to(device)).cpu().detach().numpy()
        x.append(latent[:, 0])
        y.append(latent[:, 1])
        labels.append(label)

    df = pd.DataFrame({"x": np.squeeze(np.array(x)),
                       "y": np.squeeze(np.array(y)),
                       "label": np.squeeze(np.array(labels))})
    fig, ax = plt.subplots()

    for i, dff in df.groupby("label"):
        ax.scatter(dff['x'], dff['y'], s=50, label=classes[i])
    ax.legend()
    plt.show()


def train_step(model, sample, device, optimizer, loss_fn):
    x, _ = sample
    x.requires_grad = True
    x = x.to(device)

    optimizer.zero_grad()

    x_pred = model(x)
    loss = loss_fn(x_pred, x)
    loss.backward()

    optimizer.step()
    return loss.item()


def vae_train_step(vae_model, sample, device, optimizer, loss_fn):
    x, _ = sample
    x.requires_grad = True
    x = x.to(device)

    optimizer.zero_grad()

    x_pred = vae_model(x)
    loss = loss_fn(x_pred, x) + vae_model.kl_divergence().sum()
    loss.backward()

    optimizer.step()
    return loss.item()


def train_single_batch(model, epoch_count, train_data_loader, device, train_step_fn=train_step):
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.95, 0.99))

    for epoch in range(epoch_count):
        total_loss = 0
        sample = next(iter(train_data_loader))
        loss = train_step_fn(model, sample, device, optimizer, loss_fn)
        total_loss += loss

        sys.stdout.write(
            f'\r[Epoch: {epoch + 1}/{epoch_count}]'
            f' loss: {total_loss:.3f}'
        )


def train(model, epoch_count, train_data_loader, device, train_step_fn=train_step):
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.95, 0.99))

    for epoch in range(epoch_count):
        print()
        total_loss = 0
        for i, sample in enumerate(train_data_loader, 0):
            loss = train_step_fn(model, sample, device, optimizer, loss_fn)
            total_loss += loss

            if i % 20 == 0 or i == len(train_data_loader) - 1:
                sys.stdout.write(
                    f'\r[Epoch: {epoch + 1}/{epoch_count}, Iter:{i + 1:5d}/{len(train_data_loader)}]'
                    f' batch loss: {loss:.3f}'
                    f' total loss: {total_loss / (i + 1):.3f}'
                )
        checkpoint_path = f'./assets/checkpoint_{epoch + 1}.pt'
        print()
        print(f'Saving checkpoint to {checkpoint_path}')
        torch.save(model, checkpoint_path)
        print("-" * 100)


def try_out(model, images, device):
    results = []
    for img in images:
        img = img.to(device)
        decoded = model(torch.unsqueeze(img, dim=0))
        result = torch.concat([img, torch.squeeze(decoded, dim=0)], dim=2)
        result = torch.permute(result, dims=(1, 2, 0)).cpu().detach().numpy()
        results.append(result)
    pack_size = 16
    for i in range(0, len(results), pack_size):
        result = np.concatenate(results[i: i + pack_size], axis=0)
        plt.imshow(result)
        plt.show()
