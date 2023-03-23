import sys

import torch
from matplotlib import pyplot as plt
from torch import nn, optim


def train_step(model, sample, device, optimizer, loss_fn):
    x, y = sample
    x = x.to(device)
    y = y.to(device)

    y_pred = model(x)

    optimizer.zero_grad()
    loss = loss_fn(y_pred[:, :, :y.size(2)], y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(*, model, epoch_count,
          train_data_loader, val_data_loader, test_data_loader,
          device, train_step_fn=train_step,
          checkpoint_path=None):
    loss_fn = nn.L1Loss(reduction='sum')
    optimizer = optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    model.train()

    for epoch in range(epoch_count):
        print()
        total_loss = 0
        for i, sample in enumerate(train_data_loader, 0):
            loss = train_step_fn(model, sample, device, optimizer, loss_fn)
            total_loss += loss

            sys.stdout.write(
                f'\r[Epoch: {epoch + 1}/{epoch_count}, Iter:{i + 1:5d}/{len(train_data_loader)}]'
                f' batch loss: {loss:.3f}'
                f' total loss: {total_loss / (i + 1):.3f}'
            )
        print()
        print("-" * 100)
        print("Validating model...")
        model.eval()
        if checkpoint_path is not None:
            torch.save(model, checkpoint_path)
            print(f"Saving model to[{checkpoint_path}]")

        eval_on_data(val_data_loader, model, device, loss_fn)
        model.train()
    print()
    print("-" * 100)
    print("Testing model...")
    model.eval()
    eval_on_data(test_data_loader, model, device, loss_fn)
    model.train()
    return model


def eval_on_data(eval_data_loader, model, device, loss_fn):
    total_loss = 0.0

    for i, (x, y) in enumerate(eval_data_loader):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)

        loss = loss_fn(y_pred[:, :, :y.size(2)], y)
        total_loss += loss
    print(f"Avg. loss: {total_loss / len(eval_data_loader):.3f}")

