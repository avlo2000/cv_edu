from typing import List

import torch
import abc

from rf_prop.distmap import DistMap


class Source(abc.ABC):
    @abc.abstractmethod
    def dists(self, x_mesh: torch.Tensor, y_mesh: torch.Tensor) -> DistMap:
        pass


class PointSource(Source):
    def __init__(self, loc: torch.Tensor):
        self.loc = loc
        self.reflectors: List = []

    def add_reflector(self, ref):
        self.reflectors.append(ref)

    def dists(self, x_mesh: torch.Tensor, y_mesh: torch.Tensor) -> DistMap:
        d_x = x_mesh - self.loc[0]
        d_y = y_mesh - self.loc[1]
        bright_mask = torch.ones_like(x_mesh)
        for ref in self.reflectors:
            bright_mask *= ~ref.dark_zone(self, x_mesh, y_mesh)
        points = torch.stack([d_x, d_y])
        ds = torch.linalg.norm(points, dim=0)[None, ...]
        return DistMap(dist=ds, mask=bright_mask)


class SegmentSource(Source):
    def __init__(self, p0: torch.Tensor, p1: torch.Tensor):
        self.p0 = p0[:, None, None]
        self.p1 = p1[:, None, None]

    def dists(self, x_mesh: torch.Tensor, y_mesh: torch.Tensor) -> torch.Tensor:
        p_mesh = torch.stack([x_mesh, y_mesh], dim=0)
        l2 = torch.linalg.norm(self.p1 - self.p0) ** 2
        dp0 = p_mesh - self.p0
        dp1 = self.p1 - self.p0
        t = torch.sum(dp0 * dp1, dim=0) / l2
        t = torch.clip(t, 0.0, 1.0)
        prj = self.p0 + t * (self.p1 - self.p0)
        return torch.linalg.norm(prj - p_mesh, dim=0)[None, ...]
