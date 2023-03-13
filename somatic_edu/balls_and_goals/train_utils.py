import sys

import torch
from torch import nn, optim
from torchmetrics.classification import BinaryF1Score, BinaryRecall, BinaryPrecision, BinaryAccuracy


def train_step(model, sample, device, optimizer, loss_fn):
    x, y = sample
    x = x.to(device)
    y_pred = model(x)
    y = torch.unsqueeze(y, dim=1).to(device, dtype=torch.float32)

    optimizer.zero_grad()
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_single_batch(model, epoch_count, train_data_loader, device, train_step_fn=train_step):
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(epoch_count):
        total_loss = 0
        sample = next(iter(train_data_loader))
        loss = train_step_fn(model, sample, device, optimizer, loss_fn)
        total_loss += loss

        sys.stdout.write(
            f'\r[Epoch: {epoch + 1}/{epoch_count}]'
            f' loss: {total_loss/(epoch + 1):.8f}'
        )


def train(*, model, epoch_count,
          train_data_loader, val_data_loader, test_data_loader,
          device, train_step_fn=train_step,
          checkpoint_period=2):
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=0.02)

    model.eval()
    eval_on_data(test_data_loader, model, device, loss_fn)
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
        eval_on_data(val_data_loader, model, device, loss_fn)
        if epoch % checkpoint_period == 0:
            checkpoint_path = f'./assets/checkpoint_{epoch + 1}.pt'
            print()
            print(f'Saving checkpoint to {checkpoint_path}')
            torch.save(model, checkpoint_path)
        model.train()
    print()
    print("-" * 100)
    print("Testing model...")
    model.eval()
    eval_on_data(test_data_loader, model, device, loss_fn)
    model.train()


def eval_on_data(eval_data_loader, model, device, loss_fn):
    total_loss = 0.0
    all_true_preds = []
    all_preds = []
    for x, y in eval_data_loader:
        x = x.to(device)
        y_pred = model(x)
        y = torch.unsqueeze(y, dim=1).to(device, dtype=torch.float32)

        loss = loss_fn(y_pred, y)
        loss.backward()
        total_loss += loss.item()

        all_true_preds.append(y)
        all_preds.append(y_pred)
    all_true_preds = torch.flatten(torch.stack(all_true_preds))
    all_preds = torch.flatten(torch.stack(all_preds))

    accuracy_metric = BinaryAccuracy().to(device)
    f1_metric = BinaryF1Score().to(device)
    recall_metric = BinaryRecall().to(device)
    precision_metric = BinaryPrecision().to(device)
    print(f"Avg. loss: {total_loss / len(eval_data_loader):.3f}")
    print(f"Accuracy: {accuracy_metric(all_preds, all_true_preds):.3f}")
    print(f"F1 score: {f1_metric(all_preds, all_true_preds):.3f}")
    print(f"Recall: {recall_metric(all_preds, all_true_preds):.3f}")
    print(f"Precision: {precision_metric(all_preds, all_true_preds):.3f}")
