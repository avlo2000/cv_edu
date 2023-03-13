import numpy as np
import torch

from matplotlib import pyplot as plt


def plot_latent_space(model, data, device, classes):
    import pandas as pd

    x, y, labels = [], [], []
    for instance, label in data:
        latent = model.encoder(torch.unsqueeze(instance, dim=0).to(device)).cpu().detach().numpy()
        x.append(latent[:, 0])
        y.append(latent[:, 1])
        labels.append(label)

    df = pd.DataFrame({"x": np.squeeze(np.array(x)),
                       "y": np.squeeze(np.array(y)),
                       "label": np.squeeze(np.array(labels))})
    fig, ax = plt.subplots()

    for i, dff in df.groupby("label"):
        ax.scatter(dff['x'], dff['y'], s=50, label=classes[i])
    ax.legend()
    plt.show()


def try_out(model, images, device):
    results = []
    for img in images:
        img = img.to(device)
        decoded = model(torch.unsqueeze(img, dim=0))
        result = torch.concat([img, torch.squeeze(decoded, dim=0)], dim=2)
        result = torch.permute(result, dims=(1, 2, 0)).cpu().detach().numpy()
        results.append(result)
    pack_size = 16
    for i in range(0, len(results), pack_size):
        result = np.concatenate(results[i: i + pack_size], axis=0)
        plt.imshow(result)
        plt.show()
