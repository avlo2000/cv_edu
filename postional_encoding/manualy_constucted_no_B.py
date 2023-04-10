import math
from math import exp
import torch
from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns

x_dim = 80
samples_dim = 1000

A = torch.empty(samples_dim, samples_dim)
for i in range(samples_dim):
    for j in range(samples_dim):
        A[i, j] = exp(-abs(i - j) / 100)
A[A < 0.000] = 0.0
print(A)


def residual(X):
    return torch.norm(X @ X.T - A)


def pe_approx():
    position = torch.arange(samples_dim).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, x_dim, 2) * (-math.log(1000.0) / x_dim))
    mat = torch.empty(samples_dim, x_dim)
    mat[:, 0::2] = torch.sin(position * div_term)
    mat[:, 1::2] = torch.cos(position * div_term)
    mat.requires_grad = True
    return mat


def optimize():
    X = pe_approx()#samples_dim, x_dim, requires_grad=True)
    iters = 100

    opt = optim.Rprop([X], 0.1)
    for _ in range(iters):
        loss = residual(X)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(loss)
    print(X)
    print(X @ X.T)
    return X


X = optimize().detach().numpy()

plt.subplot(311)
sns.heatmap(A)
plt.subplot(312)
sns.heatmap(X @ X.T)
plt.subplot(313)
sns.heatmap(X.T)
plt.show()
