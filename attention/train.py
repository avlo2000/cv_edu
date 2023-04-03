import torch
from torch import nn
from torch.utils import data
import train_utils
import seaborn as sns
import matplotlib.pyplot as plt
import oscillation_dataset
from layer import Attention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
BATCH_SIZE = 32
EPOCH_COUNT = 5


class RNN(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.h_dim = y_dim
        self.num_layers = 2
        self.rnn = nn.RNN(x_dim, self.h_dim, self.num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.h_dim, requires_grad=True).to(x.device)
        x = torch.transpose(x, 2, 1)
        out, hn = self.rnn(x, h0.detach())
        out = torch.transpose(out, 2, 1)
        return out


def main():
    dataset, time, phases = oscillation_dataset.OscillationDataset.default(
        100_000,
        200
    )
    train_data, test_data = data.random_split(dataset, [0.7, 0.3])
    train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = Attention(dataset.x_shape[1], dataset.y_shape[1]).to(DEVICE)
    # model = RNN(dataset.x_shape[1], dataset.y_shape[1]).to(DEVICE)
    for series, _ in test_data:
        x = torch.unsqueeze(series, dim=0).to(DEVICE)
        y = torch.squeeze(model(x))
        sep = dataset.x_shape[0]
        y_size = dataset.y_shape[0]
        for ph in range(dataset.phase_count):
            plt.subplot(2, 1, 1)
            plt.plot(time[:sep], series[ph])
            plt.plot(time[sep:], y[ph, :y_size].cpu().detach(), label=f"Phase {ph}")
        plt.legend()
        plt.show()
    print(f"Params count: {sum(p.nelement() for p in model.parameters())}")
    model = train_utils.train(
                        model=model,
                        epoch_count=EPOCH_COUNT,
                        train_data_loader=train_data_loader,
                        val_data_loader=test_data_loader,
                        test_data_loader=test_data_loader,
                        device=DEVICE
                    )
    for series, _ in test_data:
        x = torch.unsqueeze(series, dim=0).to(DEVICE)
        y = torch.squeeze(model(x))
        sep = dataset.x_shape[0]
        y_size = dataset.y_shape[0]
        for ph in range(dataset.phase_count,):
            plt.subplot(2, 1, 1)
            plt.plot(time[:sep], series[ph])
            plt.plot(time[sep:], y[ph, :y_size].cpu().detach(), label=f"Phase {ph}")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
