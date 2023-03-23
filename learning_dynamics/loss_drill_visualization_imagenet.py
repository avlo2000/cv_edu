from torchvision import datasets
import torch
from torch import nn
from torch.utils import data
from torchvision.transforms import transforms
from torchvision import models

from learning_dynamics.utils import loss_drill, load_uid_to_idx
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
BATCH_SIZE = 10
EPOCH_COUNT = 50


def targets_to_imagenet_targets_fn(dataset: datasets.ImageFolder, id_to_idx):
    def transform(trg):
        uid = dataset.classes[trg]
        idx = id_to_idx[uid]
        return idx

    return transform


def main():
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    trained_model = models.efficientnet_v2_s(weights=weights).to(DEVICE)
    stupid_model = models.efficientnet_v2_s().to(DEVICE)
    # over_fitted_model = torch.load('../transfer_learning/assets/ResNet18_ImageWoof.pkl')

    data_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(
        root='../data/imagewoof2-320/val',
        transform=data_transforms
    )
    uid_to_idx = load_uid_to_idx('../data/imagewoof2-320/imagenet_class_index.json')
    trg_transform = targets_to_imagenet_targets_fn(dataset, uid_to_idx)
    dataset.target_transform = trg_transform
    sample, _ = torch.utils.data.random_split(dataset, lengths=[300, len(dataset)-300])
    test_data_loader = data.DataLoader(sample, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    weights, losses = loss_drill(trained_model, stupid_model, nn.CrossEntropyLoss(),
                                 test_data_loader, ticks_count=100, device=DEVICE)

    plt.title("Loss drill before train and after")
    plt.xlabel("ModelA to ModelB lerp weight")
    plt.ylabel("Total cross entropy loss")
    plt.plot(weights, losses, label=f"Loss drill")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
