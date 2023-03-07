import torch.utils.data

import torch.utils.data
import torchvision.transforms as transforms
from torchsummary import summary
from torchvision import datasets

from autoencoders.variational_autoencoder import VariationalAutoencoder
from autoencoders.utils import plot_latent_space, train, try_out, vae_train_step

EPOCH_COUNT = 30
BATCH_SIZE = 64
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


model = VariationalAutoencoder(in_shape=in_shape, encoded_space_dim=2).to(device)
summary(model, input_size=in_shape, device=device)


try_out(model, next(iter(test_data_loader))[0], device)
plot_latent_space(model, test_data, device, classes)
train(model, EPOCH_COUNT, train_data_loader, device, vae_train_step)
print("Testing...")
plot_latent_space(model, test_data, device, classes)
try_out(model, next(iter(test_data_loader))[0], device)
