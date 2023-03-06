import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, in_shape, encoded_space_dim):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(True),
            nn.Linear(512, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, in_shape, encoded_space_dim):
        super().__init__()
        self.in_shape = in_shape
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128 * 16 * 16),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(128, 16, 16))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_shape[0], kernel_size=3, stride=1, padding=1, output_padding=0),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        x = torch.nn.functional.interpolate(x, self.in_shape[1:])
        return x


class BigCNNAutoencoder(nn.Module):
    def __init__(self, in_shape, encoded_space_dim):
        super().__init__()
        self.encoder = Encoder(in_shape, encoded_space_dim)
        self.decoder = Decoder(in_shape, encoded_space_dim)

    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return decoded_x
