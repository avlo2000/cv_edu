import sys

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear()
        )

    def forward(self, x):
        pass