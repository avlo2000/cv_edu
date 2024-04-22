from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from rf_prop.distmap import DistMap
from rf_prop.sources import PointSource


class SegmentReflector:
    def __init__(self, p0: torch.Tensor, p1: torch.Tensor, ):
        self.p0 = p0
        self.p1 = p1
        self.auxiliary_sources: List[PointSource] = []

    def add_point_source(self, source: PointSource):
        l2 = torch.linalg.norm(self.p1 - self.p0) ** 2
        dp0 = source.loc - self.p0
        dp1 = self.p1 - self.p0
        t = torch.sum(dp0 * dp1, dim=0) / l2
        prj = self.p0 + t * (self.p1 - self.p0)
        ref = 2.0 * prj - source.loc

        src = PointSource(ref)
        self.auxiliary_sources.append(src)

    def dists(self, x_mesh: torch.Tensor, y_mesh: torch.Tensor) -> List[DistMap]:
        ds = []
        for src in self.auxiliary_sources:
            include_mask = self.reflected_zone(src, x_mesh, y_mesh)
            d = src.dists(x_mesh, y_mesh)
            d.mask = include_mask
            ds.append(d)
        return ds

    def reflected_zone(self, source: PointSource, x_mesh: torch.Tensor, y_mesh: torch.Tensor):
        p0, p1 = sort_cw(source.loc, self.p0, self.p1)
        mask_l0 = is_left_from_line(source.loc, p0, x_mesh, y_mesh)
        mask_l1 = ~is_left_from_line(source.loc, p1, x_mesh, y_mesh)
        mask_p01 = ~is_left_from_line(p0, p1, x_mesh, y_mesh)
        return mask_l0 & mask_l1 & mask_p01

    def dark_zone(self, source: PointSource, x_mesh: torch.Tensor, y_mesh: torch.Tensor):
        p0, p1 = sort_cw(source.loc, self.p0, self.p1)
        mask_l0 = is_left_from_line(source.loc, p0, x_mesh, y_mesh)
        mask_l1 = ~is_left_from_line(source.loc, p1, x_mesh, y_mesh)
        mask_p01 = ~is_left_from_line(p0, p1, x_mesh, y_mesh)
        return mask_l0 & mask_l1 & mask_p01


def cat_zero_to_3dim(p: torch.Tensor):
    return torch.cat([p, torch.zeros_like(p[0][None, ...])])


def is_left_from_line(p0: torch.Tensor, p1: torch.Tensor, x_mesh: torch.Tensor, y_mesh: torch.Tensor):
    pp = torch.stack([x_mesh, y_mesh])
    v0 = (p1 - p0)[:, None, None]
    v1 = pp - p1[:, None, None]
    v0 = cat_zero_to_3dim(v0)
    v1 = cat_zero_to_3dim(v1)
    cross_z = torch.cross(v1, v0, dim=0)[2]
    return cross_z > 0.0


def sort_cw(ref_p: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    v0 = p0 - ref_p
    v1 = p1 - ref_p
    v0 = cat_zero_to_3dim(v0)
    v1 = cat_zero_to_3dim(v1)
    cross_z = torch.cross(v1, v0, dim=0)[2]
    if torch.all(cross_z > 0.0):
        return p0, p1
    return p1, p0


if __name__ == '__main__':
    def main():
        x_lin = torch.linspace(-50, 50, 600)
        y_lin = torch.linspace(-50, 50, 600)
        x_mesh, y_mesh = torch.meshgrid(x_lin, y_lin, indexing='xy')
        source = PointSource(torch.tensor([0.0, 0.0]))
        ref = SegmentReflector(torch.tensor([-20, -20.0]), torch.tensor([-20, 20.0]))
        ref.add_point_source(source)
        canva = ref.dark_zone(source, x_mesh, y_mesh).numpy().squeeze()

        import cv2
        cv2.imshow('canva', canva.astype(np.uint8) * 255)
        cv2.waitKey()


    main()
