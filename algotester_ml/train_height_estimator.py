from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils import data

from algotester_ml.height_estimator import HeightEstimator
from algotester_ml.load_data import load_data, HeightEstimationDataset
import pytorch_lightning as pl


class PLHeightEstimator(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model = HeightEstimator()

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        (img, mask), h = batch
        h_hat = self.model(img, mask)
        loss = nn.functional.mse_loss(h_hat, h)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    dataset = HeightEstimationDataset(Path('data/mlc_training_data'))
    train, val = data.random_split(dataset, [0.8, 0.2])

    model = PLHeightEstimator()
    trainer = pl.Trainer()
    trainer.fit(model, data.DataLoader(train), data.DataLoader(val))


if __name__ == '__main__':
    main()
