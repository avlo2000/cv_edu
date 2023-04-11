import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(20000.0) / d_model))
        mat = torch.zeros(1, max_len, d_model)
        mat[0, :, 0::2] = torch.sin(position * div_term)
        mat[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('mat', mat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mat[:, :x.size(1), :]
        return x


if __name__ == '__main__':
    def main():
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        pe = PositionalEncoding(256, 512)
        mat = np.squeeze(pe.mat.numpy())
        plt.subplot(311)
        sns.heatmap(mat)
        plt.subplot(312)
        sns.heatmap(mat @ mat.T)
        plt.subplot(313)
        sns.heatmap(mat.T @ mat)
        plt.show()

    main()
