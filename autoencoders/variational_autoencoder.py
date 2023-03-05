import torch
from torch import nn


class VariationalAutoencoder(nn.Module):
    def __init__(self, in_shape, encoded_space_dim):
        super().__init__()
        self.input_shape = in_shape
        self.encoder_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_shape.numel(), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.encoder_sigma = nn.Linear(256, encoded_space_dim)
        self.encoder_mean = nn.Linear(256, encoded_space_dim)
        self.decoder = nn.Sequential(
            nn.Linear(encoded_space_dim, 512),
            nn.ReLU(),
            nn.Linear(512, in_shape.numel()),
        )

        self.distribution = torch.distributions.Normal(0, 1)

    def forward(self, x):
        x = self.encoder_head(x)
        sigma = self.encoder_sigma(x)
        mean = self.encoder_mean(x)
        encoded_x = mean + sigma * self.distribution.sample(mean.shape)
        decoded_x = self.decoder(encoded_x)
        return torch.reshape(decoded_x, (x.size(0), *self.input_shape))
