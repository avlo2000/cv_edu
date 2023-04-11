import torch
from torch import nn, optim
from postional_encoding.utils import cross_sample_correlations_goal


class TunedEncoding(nn.Module):
    ITERS_TO_SOLVE = 200

    def __init__(self, d_model: int, max_len: int = 5000, temperature=100.0, verbose=0):
        super().__init__()
        self.verbose = verbose
        self.temperature = temperature
        self.d_model = d_model
        self.max_len = max_len
        mat = self.__optimize_positional_correlations()
        mat = torch.unsqueeze(mat, dim=0)

        self.register_buffer('mat', mat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mat[:, :x.size(1), :] / 100.0
        return x

    def __optimize_positional_correlations(self):
        mat = torch.rand(self.max_len, self.d_model, requires_grad=True)

        iters = 100

        corr_gaol = cross_sample_correlations_goal(self.max_len, self.temperature)
        opt = optim.Rprop([mat], 0.1)
        for _ in range(iters):
            loss = self.__residual_corr(mat, corr_gaol)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if self.verbose == 1:
                print(loss)
        return mat

    @staticmethod
    def __residual_corr(mat, corr_gaol):
        return torch.norm(mat @ mat.T - corr_gaol) + 0.1 * torch.norm(mat.T @ mat)


if __name__ == '__main__':
    def main():
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        pe = TunedEncoding(512, 1500, verbose=1)
        mat = np.squeeze(pe.mat.detach().numpy())
        plt.subplot(311)
        sns.heatmap(mat)
        plt.subplot(312)
        sns.heatmap(mat @ mat.T)
        plt.subplot(313)
        sns.heatmap(mat.T @ mat)
        plt.show()

    main()
