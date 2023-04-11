import math

import torch
from torch import nn

torch.random.manual_seed(0)


class Attention(nn.Module):
    def __init__(self, dx, dq, dk=None, dv=None):
        super().__init__()
        self.dq = dq
        dk = dq if dk is None else dk
        dv = dq if dv is None else dv
        self.Wq = nn.Parameter(torch.rand(dq, dx, dtype=torch.float32))
        self.Wk = nn.Parameter(torch.rand(dk, dx, dtype=torch.float32))
        self.Wv = nn.Parameter(torch.rand(dv, dx, dtype=torch.float32))

    def forward(self, x):
        Q = self.Wq @ x
        K = self.Wk @ x
        V = self.Wv @ x

        scores = Q @ torch.transpose(K, 2, 1)
        normed_scores = torch.softmax(scores / math.sqrt(self.dq), dim=1)
        attention = normed_scores @ V
        return attention

    def scores(self, x):
        Q = self.Wq @ x
        K = self.Wk @ x

        scores = Q @ torch.transpose(K, 2, 1)
        normed_scores = torch.softmax(scores / math.sqrt(self.dq), dim=1)
        return normed_scores


if __name__ == '__main__':
    model = nn.Sequential(
        Attention(100, 64),
    )
