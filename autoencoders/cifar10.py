import torch.utils.data
from torchsummary import summary
from torchvision import transforms, datasets

from autoencoders.big_cnn_autoencoder import BigCNNAutoencoder
from autoencoders.pretrained_autoencoder import PretrainedAutoencoder
from autoencoders.utils import plot_latent_space, train, try_out

EPOCH_COUNT = 10
BATCH_SIZE = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
data_transforms = transforms.Compose([transforms.ToTensor()])

train_data = datasets.CIFAR10(
    root='../data',
    train=True,
    transform=data_transforms,
    download=True,
)
test_data = datasets.CIFAR10(
    root='../data',
    train=False,
    transform=data_transforms
)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
in_shape: torch.Size = next(iter(train_data_loader))[0].shape


# model = BigCNNAutoencoder(in_shape=in_shape[1:], encoded_space_dim=16).to(device)
model = PretrainedAutoencoder().to(device)
state_dict = torch.load('./assets/chenjie/autoencoder.pkl')
model.load_state_dict(state_dict)
summary(model, input_size=in_shape[1:], device=device)


# train(model, EPOCH_COUNT, train_data_loader, device)
try_out(model, next(iter(test_data_loader))[0], device)
# plot_latent_space(model, test_data, device, test_data.classes)

torch.linalg.tensorinv