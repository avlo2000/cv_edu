import torch
from torch import nn
from torch.utils import data
import train_utils
import matplotlib.pyplot as plt
import oscillation_dataset
from layer import Attention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
BATCH_SIZE = 64
EPOCH_COUNT = 5


def main():
    dataset = oscillation_dataset.OscillationDataset.default(
        100_000,
        200
    )
    train_data, test_data = data.random_split(dataset, [0.7, 0.3])
    train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = Attention(dataset.x_shape[1], dataset.y_size[1]).to(DEVICE)
    model = train_utils.train(
                        model=model,
                        epoch_count=EPOCH_COUNT,
                        train_data_loader=train_data_loader,
                        val_data_loader=test_data_loader,
                        test_data_loader=test_data_loader,
                        device=DEVICE
                    )


if __name__ == '__main__':
    main()
