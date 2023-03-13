import sys

import torch
from torch import nn, optim


def train_step(model, sample, device, optimizer, loss_fn):
    x, _ = sample
    x = x.to(device)

    optimizer.zero_grad()

    x_pred = model(x)
    loss = loss_fn(x_pred, x)
    loss.backward()

    optimizer.step()
    return loss.item()


def vae_train_step(vae_model, sample, device, optimizer, loss_fn):
    x, _ = sample
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
