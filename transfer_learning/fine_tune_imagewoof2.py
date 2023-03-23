import torch
import torch.utils.data as data

from torch import nn
from torchvision import datasets, transforms, models

import data_preparation
import train_utils
from torchsummary import summary

from learning_dynamics.utils import load_uid_to_idx


def targets_to_imagenet_targets_fn(dataset: datasets.ImageFolder, id_to_idx):
    def transform_fn(trg):
        uid = dataset.classes[trg]
        idx = id_to_idx[uid]
        return idx

    return transform_fn


EPOCH_COUNT = 5
BATCH_SIZE = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
torch.cuda.empty_cache()

weights = models.EfficientNet_V2_S_Weights.DEFAULT
pretrained_model = models.efficientnet_v2_s(weights=weights).to(device)
model_classes = weights.meta["categories"]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
train_data = data_preparation.load_imagewoof_train(transform)
val_data = data_preparation.load_imagewoof_val(transform)
uid_to_idx = load_uid_to_idx('../data/imagewoof2-320/imagenet_class_index.json')
trg_transform = targets_to_imagenet_targets_fn(val_data, uid_to_idx)
train_data.target_transform = trg_transform
val_data.target_transform = trg_transform


train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_data_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

print(f"Train size: {len(train_data)} \n"
      f"Val size: {len(val_data)}")

model = train_utils.train(
    model=pretrained_model,
    epoch_count=EPOCH_COUNT,
    train_data_loader=train_data_loader,
    test_data_loader=val_data_loader,
    val_data_loader=val_data_loader,
    device=device,
    checkpoint_path='./assets/EfficientNet_V2_S_ImageWoof.pkl'
)

