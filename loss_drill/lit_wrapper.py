from typing import Any, Optional

import lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch import optim
from torchmetrics import Accuracy


class LightningModel(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.train_accuracy_metric = Accuracy("multiclass")
        self.val_accuracy_metric = Accuracy("multiclass")
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)

        loss = self.loss_fn(y, y_pred)
        accuracy = self.accuracy_metric(y_pred, y)

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)

        loss = self.loss_fn(y_pred, y)
        accuracy = self.accuracy_metric(y_pred, y)

        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
