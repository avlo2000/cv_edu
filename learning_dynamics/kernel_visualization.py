import matplotlib.pyplot as plt
import torch
from torchvision import datasets, models, utils


def load_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    return model


def main():
    model = load_model()

    kernels = model.conv1.weight.cpu().detach().clone()
    # kernels = model.features[0][0].weight.cpu().detach().clone()
    # kernels = model.features[0].weight.cpu().detach().clone()

    print(kernels.shape)
    kernels_grid = torch.permute(utils.make_grid(kernels), dims=(1, 2, 0))
    plt.imshow(kernels_grid.numpy())
    plt.show()


if __name__ == '__main__':
    main()
