import torch


def elliptic_fit2d(points2d: torch.Tensor):
    rhs_mat = torch.stack([
        points2d[:, 0]**2,
        -2 * points2d[:, 0],
        points2d[:, 1] ** 2,
        -2 * points2d[:, 1],
        torch.ones_like(points2d[:, 0])
    ],)
    lhs_mat = torch.ones_like(points2d[:, 0])