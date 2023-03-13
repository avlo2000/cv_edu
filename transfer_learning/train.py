import torch
import torch.utils.data as data

import torchvision
from torch import nn
from torchvision.transforms import transforms

import data_preparation
import train_utils
from torchsummary import summary

from transfer_learning.simple_cnn import SimpleCNN

EPOCH_COUNT = 20
BATCH_SIZE = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
torch.cuda.empty_cache()

pretrained_model = torchvision.models.efficientnet_v2_s(weight=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1) \
    .to(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.ToTensor(),
])
train_data, test_data, val_data = data_preparation.load_dtd(transform)

train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_data_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

print(f"Train size: {len(train_data)} \n"
      f"Test size: {len(test_data)} \n"
      f"Val size: {len(val_data)}")

model = pretrained_model
model.classifier[1] = nn.Linear(in_features=1280, out_features=len(test_data.dataset.classes)).to(device)

# model.fc = nn.Linear(model.fc.in_features, len(test_data.dataset.classes)).to(device)
# model = SimpleCNN(len(test_data.dataset.classes)).to(device)
summary(model, next(iter(train_data))[0].shape, device=device)

# train_utils.train_single_batch(model, 1000, train_data_loader, device)
train_utils.train(
    model=model,
    epoch_count=EPOCH_COUNT,
    train_data_loader=train_data_loader,
    test_data_loader=test_data_loader,
    val_data_loader=val_data_loader,
    device=device
)
