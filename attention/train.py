import torch
from torch import nn
from torch.utils import data
import train_utils
import matplotlib.pyplot as plt
import oscillation_dataset
import attention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
BATCH_SIZE = 64
EPOCH_COUNT = 200


def main():
    dataset = oscillation_dataset.OscillationDataset.default(
        100_000,
        200
    )
    train_data, test_data = data.random_split(dataset, [0.7, 0.3])
    train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = nn.Sequential(
        nn.Linear(dataset.x_size(), 1024),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1024, dataset.y_size()),
    ).to(DEVICE)
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
