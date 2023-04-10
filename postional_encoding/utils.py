import math

import torch


def cross_sample_correlations_goal(dim, temperature):
    x, y = torch.meshgrid(torch.arange(0, dim), torch.arange(0, dim), indexing='xy')
    return torch.exp(-torch.abs(x - y) / temperature)


def cross_feature_correlations_goal(dim):
    return torch.eye(dim)
