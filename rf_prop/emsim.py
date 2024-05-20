from typing import List

import matplotlib.pyplot as plt
import torch

from rf_prop.distmap import DistMap
from rf_prop.reflectors import SegmentReflector
from rf_prop.sources import PointSource


class EMSim:
    def __init__(self,
                 x_min: float, x_max: float, x_resolution: int,
                 y_min: float, y_max: float, y_resolution: int):
        x_lin = torch.linspace(x_min, x_max, x_resolution)
        y_lin = torch.linspace(y_min, y_max, y_resolution)
        self.x_mesh, self.y_mesh = torch.meshgrid(x_lin, y_lin, indexing='xy')
        self.sources = []
        self.reflectors = []

    def add_reflector(self, ref: SegmentReflector):
        self.reflectors.append(ref)

    def add_point_source(self, src: PointSource):
        self.sources.append(src)

    def calculate_distmaps(self) -> (torch.Tensor, torch.Tensor):
        dists = []
        masks = []
        for ref in self.reflectors:
            for src in self.sources:
                src.add_reflector(ref)
                ref.add_point_source(src)
        for ref in self.reflectors:
            dmaps = ref.dists(self.x_mesh, self.y_mesh)
            dists.extend(dmap.dist.squeeze() for dmap in dmaps)
            masks.extend(dmap.mask for dmap in dmaps)
        for src in self.sources:
            dmap = src.dists(self.x_mesh, self.y_mesh)
            dists.append(dmap.dist.squeeze())
            masks.append(dmap.mask)
        dists = torch.stack(dists, dim=0)
        masks = torch.stack(masks, dim=0)
        return dists, masks
