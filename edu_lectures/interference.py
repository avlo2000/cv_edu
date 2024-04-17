import matplotlib.pyplot as plt
import torch
import cv2

d = 0.1
wave_speed = 1.6
loc = torch.tensor([
    [0.0, 0.0],
]).T
decay = torch.tensor(10.0)
freq = torch.tensor(5)


def wave_front(p: torch.Tensor, dists: torch.Tensor, time: float) -> torch.Tensor:
    mask = dists > time * wave_speed
    for i in range(mask.shape[-1]):
        p[mask[:, :, i]] = 0.0
    return p


def potential(x: torch.Tensor, y: torch.Tensor, time: float) -> torch.Tensor:
    x = loc[0].repeat(*x.shape, 1) - x[..., None]
    y = loc[1].repeat(*y.shape, 1) - y[..., None]
    points = torch.stack([x, y])
    dists = 2.0 * torch.pi * freq * torch.linalg.norm(points, dim=0)
    p = torch.sum(torch.sin(dists - time) / torch.sqrt(dists), dim=-1)
    return wave_front(p, dists, time)


def main():
    x_lin = torch.linspace(-5, 5, 600)
    y_lin = torch.linspace(-5, 5, 600)
    x_mesh, y_mesh = torch.meshgrid(x_lin, y_lin, indexing='xy')

    for time in torch.linspace(0.0, 6000, 900):
        z = potential(x_mesh, y_mesh, time)
        cv2.imshow("waves", z.numpy())
        cv2.waitKey(20)
        # plt.imshow(z, cmap='hot', interpolation='nearest')
        # plt.show()


if __name__ == '__main__':
    main()
