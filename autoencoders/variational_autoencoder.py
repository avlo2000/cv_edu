import torch
from torch import nn


class VariationalEncoder(nn.Module):
    def __init__(self, in_shape, encoded_space_dim):
        super().__init__()
        self.encoder_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_shape.numel(), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.encoder_sigma = nn.Linear(128, encoded_space_dim)
        self.encoder_mean = nn.Linear(128, encoded_space_dim)
        self.distribution = torch.distributions.Normal(0, 1)
        self.sigma = torch.empty(encoded_space_dim)
        self.mean = torch.empty(encoded_space_dim)
        self.__eps = torch.scalar_tensor(0.0001)

    def forward(self, x):
        x = self.encoder_head(x)
        self.sigma = torch.sigmoid(self.encoder_sigma(x))
        self.mean = torch.sigmoid(self.encoder_mean(x))
        encoded_x = self.mean + self.sigma * self.distribution.sample(self.mean.shape).to(self.mean.device)
        return encoded_x

    def kl_divergence(self):
        kl_div = -torch.log(self.sigma + self.__eps) + self.sigma ** 2 + self.mean ** 2 / 2 - 0.5
        return kl_div


class VariationalAutoencoder(nn.Module):
    def __init__(self, in_shape, encoded_space_dim):
        super().__init__()
        self.input_shape = in_shape
        self.encoder = VariationalEncoder(in_shape, encoded_space_dim)
        self.decoder = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, in_shape.numel()),
        )

    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        decoded_x = torch.reshape(decoded_x, (x.size(0), *self.input_shape))
        return decoded_x

    def kl_divergence(self):
        return self.encoder.kl_divergence()
