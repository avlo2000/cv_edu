from torchvision import datasets
import torch
from torch import nn
from torch.utils import data
from torchvision.transforms import transforms
from torchvision import models

from learning_dynamics.utils import loss_drill, load_clsloc
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
BATCH_SIZE = 16
EPOCH_COUNT = 50


def reduce_targets(dataset: datasets.ImageFolder, id_to_cls, cls_to_idx):
    # model_classes = [cls.lower().replace(" ", "_") for cls in model_classes]
    # cls_to_idx = {cls: idx for idx, cls in enumerate(model_classes)}
    targets = []
    for trg in dataset.targets:
        uid = dataset.classes[trg]
        cls = id_to_cls[uid].lower()
        print(cls)
        idx = cls_to_idx[cls]
        targets.append(idx)
    dataset.class_to_idx = cls_to_idx
    dataset.classes = list(cls_to_idx.keys())
    dataset.targets = targets
    return dataset


def main():
    weights = models.VGG19_Weights.IMAGENET1K_V1
    model_classes = weights.meta["categories"]
    trained_model = models.vgg19(weights=weights).to(DEVICE)
    stupid_model = models.vgg19().to(DEVICE)

    data_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(
        root='../data/imagewoof2-320/val',
        transform=data_transforms
    )
    cls_to_idx, id_to_cls = load_clsloc('../data/imagewoof2-320/map_clsloc.txt')
    dataset = reduce_targets(dataset, id_to_cls, cls_to_idx)

    sample, _ = torch.utils.data.random_split(dataset, lengths=[50, len(dataset)-50])
    test_data_loader = data.DataLoader(sample, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    weights, losses = loss_drill(trained_model, stupid_model, nn.CrossEntropyLoss(),
                                 test_data_loader, ticks_count=70, device=DEVICE)

    plt.title("Loss drill before train and after")
    plt.xlabel("ModelA to ModelB lerp weight")
    plt.ylabel("Total cross entropy loss")
    plt.plot(weights, losses, label=f"Loss drill num")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
