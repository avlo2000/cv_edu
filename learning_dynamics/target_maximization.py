import sys

from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import models, utils


def load_vit_model():
    weights = models.ViT_B_32_Weights.IMAGENET1K_V1
    model = models.vit_b_32(weights=weights)
    model.eval()
    categories = weights.meta["categories"]
    return model, weights.transforms(), categories


def load_efficient_model():
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.eval()
    categories = weights.meta["categories"]
    return model, weights.transforms(), categories


def load_vgg_model():
    weights = models.VGG19_Weights.IMAGENET1K_V1
    model = models.vgg19(weights=weights)
    model.eval()
    categories = weights.meta["categories"]
    return model, weights.transforms(), categories


def pgd_adversarial(x_initial: torch.Tensor, y_trg: torch.Tensor, model: nn.Module,
                    steps=100, regularization_factor=10000.0, verbose=False):
    x_adv = torch.clone(x_initial)
    y_loss_fn = nn.CrossEntropyLoss()
    x_loss_fn = nn.MSELoss()
    opt = optim.SGD([x_adv], 0.01)
    if verbose:
        print()
    for i in range(steps):
        x_adv.requires_grad = True
        y_pred = model(torch.sigmoid(x_adv))
        loss = y_loss_fn(y_pred, y_trg) + regularization_factor * x_loss_fn(x_adv, x_initial)
        loss.backward()
        if verbose:
            sys.stdout.write(f"\rIter [{i}] loss [{loss.item()}]")
        opt.step()
    if verbose:
        print('\n')
    return torch.sigmoid(x_initial), torch.sigmoid(x_adv)


def predict(model, categories, x, top_k=1):
    probs = model(x)
    probs = torch.softmax(probs, dim=1)
    print(f"{model.__class__.__name__} prediction:")
    probs, indices = torch.topk(probs, top_k)
    for prob, idx in zip(torch.squeeze(probs), torch.squeeze(indices)):
        print(f"     {categories[idx]}-{prob.item() * 100:.3f}%")


def main():
    model, preprocess, categories = load_efficient_model()
    vgg_model, vgg_preprocess, _ = load_vgg_model()

    y_trg = torch.tensor([0.0]*1000)
    trg_idx = 0
    print(f"\nTarget class: {categories[trg_idx]}")
    y_trg[trg_idx] = 1.0
    y_trg = torch.unsqueeze(y_trg, dim=0)
    # x_initial = torch.normal(0, 1, size=(1, 3, 224, 224))
    x_initial = torch.ones(size=(1, 3, 224, 224))
    x_initial, x_adv = pgd_adversarial(x_initial, y_trg, vgg_model, steps=400, regularization_factor=0.0, verbose=True)

    print("ADVERSARIAL")
    predict(model, categories, x_adv, top_k=5)
    predict(vgg_model, categories, x_adv, top_k=5)

    grid = utils.make_grid([torch.squeeze(x_initial), torch.squeeze(x_adv)])
    plt.imshow(torch.permute(grid, dims=(1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    main()
