import torch.utils.data

import torchvision.transforms as transforms
from torch import nn
from torchsummary import summary
from torchvision import datasets
from collections import Counter
from somatic_edu.balls_and_goals.train_utils import train, train_single_batch

EPOCH_COUNT = 100
BATCH_SIZE = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

data_transforms = transforms.Compose([transforms.ToTensor(),
                                      ])

dataset = datasets.ImageFolder(
    root='./data_augmented_x1000',
    transform=data_transforms,
)
train_data, test_data = torch.utils.data.random_split(dataset, (0.8, 0.2))
train_classes = [label for _, label in train_data]
test_classes = [label for _, label in test_data]
print(Counter(train_classes))
print(Counter(test_classes))

print(dataset.class_to_idx)
print(len(train_data))
print(len(test_data))


train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
in_shape: torch.Size = train_data[0][0].shape

model = nn.Sequential(
    nn.Conv2d(in_shape[0], 8, kernel_size=3, padding=1, stride=2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(p=0.3),

    nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
    nn.Dropout(p=0.3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 128),
    nn.Dropout(p=0.3),
    nn.ReLU(),

    nn.Linear(128, 1),
    nn.Sigmoid(),
)

summary(model, input_size=in_shape, device=device)

train(model=model,
      epoch_count=EPOCH_COUNT,
      train_data_loader=train_data_loader,
      val_data_loader=test_data_loader,
      test_data_loader=test_data_loader,
      device=device)
