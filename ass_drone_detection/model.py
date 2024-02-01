import torch
from torch import nn
from torchvision.models import SwinTransformer


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()


class PredictionHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, x_hidden: torch.Tensor) -> torch.Tensor:
        pass
