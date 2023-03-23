import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

from learning_dynamics import models
from transfer_learning import train_utils


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


def select_by_class(dataset, classes):
    class_ids = [dataset.class_to_idx[cls] for cls in classes]
    indices = [i for i, label in enumerate(dataset.targets) if label in class_ids]
    subset = data.Subset(dataset, indices)
    return subset


def train_model(dataset, model, device):
    batch_size = 64
    epoch_count = 5
    train_data, test_data = data.random_split(dataset, [0.7, 0.3])
    train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
    trained_model = train_utils.train(
                        model=model,
                        epoch_count=epoch_count,
                        train_data_loader=train_data_loader,
                        val_data_loader=test_data_loader,
                        test_data_loader=test_data_loader,
                        device=device
                    )
    return trained_model


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    classes_a = dataset.classes[:5]
    classes_b = dataset.classes[5:]
    print(f"Classes for model A: {classes_a}")
    print(f"Classes for model B: {classes_b}")

    subset_a = select_by_class(dataset, classes_a)
    subset_b = select_by_class(dataset, classes_b)

    in_shape: torch.Size = dataset[0][0].shape

    model_a = models.create_model_bigger_cnn(in_shape)
    model_b = models.create_model_bigger_cnn(in_shape)

    print('='*100)
    print("Training model A")
    model_a = train_model(subset_a, model_a, DEVICE)
    torch.save(model_a, "./assets/cifar10_bigger_cnn_A.pkl")

    print('='*100)
    print("Training model B")
    model_b = train_model(subset_b, model_b, DEVICE)
    torch.save(model_b, "./assets/cifar10_bigger_cnn_B.pkl")


if __name__ == '__main__':
    main()
