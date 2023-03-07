import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, in_shape, encoded_space_dim):
        super().__init__()
        self.input_shape = in_shape
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_shape.numel(), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, encoded_space_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_space_dim, 512),
            nn.ReLU(),
            nn.Linear(512, in_shape.numel()),
        )

    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return torch.reshape(decoded_x, (x.size(0), *self.input_shape))
