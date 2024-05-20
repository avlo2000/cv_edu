import torch


class DistMap:
    def __init__(self, dist: torch.Tensor, mask: torch.Tensor):
        self.dist = dist
        self.mask = mask
