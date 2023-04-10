import math
import torch
from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns

from postional_encoding.utils import cross_sample_correlations_goal, cross_feature_correlations_goal

x_dim = 512
samples_dim = 2000

A = cross_sample_correlations_goal(samples_dim, 100)
B = cross_feature_correlations_goal(x_dim)


def residual(X):
    return torch.norm(X @ X.T - A)# + 0.01 * torch.norm(X.T @ X - B)


def pe_approx():
    position = torch.arange(samples_dim).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, x_dim, 2) * (-math.log(1000.0) / x_dim))
    mat = torch.empty(samples_dim, x_dim)
    mat[:, 0::2] = torch.sin(position * div_term)
    mat[:, 1::2] = torch.cos(position * div_term)
    mat.requires_grad = True
    return mat


def optimize():
    #X = pe_approx()
    X = torch.eye(samples_dim, x_dim, requires_grad=True)
    iters = 100

    opt = optim.Rprop([X], 0.1)
    for _ in range(iters):
        loss = residual(X)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(loss)
    return X


X = optimize().detach().numpy()

plt.subplot(411)
sns.heatmap(A)
plt.subplot(412)
sns.heatmap(X @ X.T)
plt.subplot(413)
sns.heatmap(X.T @ X)
plt.subplot(414)
sns.heatmap(X.T)
plt.show()
