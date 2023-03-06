import torch.utils.data

import torch.utils.data
import torchvision.transforms as transforms
from torchsummary import summary
from torchvision import datasets

from autoencoders.autoencoder import Autoencoder
from autoencoders.big_cnn_autoencoder import BigCNNAutoencoder
from autoencoders.cnn_autoencoder import CNNAutoencoder
from autoencoders.utils import plot_latent_space, train, try_out

EPOCH_COUNT = 10
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
classes = tuple(map(str, range(10)))

data_transforms = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST(
    root='../data',
    train=True,
    transform=data_transforms,
    download=True,
)
test_data = datasets.MNIST(
    root='../data',
    train=False,
    transform=data_transforms
)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
in_shape: torch.Size = train_data[0][0].shape


model = CNNAutoencoder(in_shape=in_shape, encoded_space_dim=2).to(device)
summary(model, input_size=in_shape, device=device)

try_out(model, next(iter(test_data_loader))[0], device)
train(model, EPOCH_COUNT, train_data_loader, device)
print("Testing...")
plot_latent_space(model, test_data, device, classes)
try_out(model, next(iter(test_data_loader))[0], device)
