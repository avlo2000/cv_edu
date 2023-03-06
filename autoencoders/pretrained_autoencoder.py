from torch import nn


class PretrainedAutoencoder(nn.Module):
    def __init__(self):
        super(PretrainedAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=3, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 32, 3, stride=3, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

