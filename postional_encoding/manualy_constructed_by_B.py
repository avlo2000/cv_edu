from cmath import exp

import torch
import torch.optim as optim
torch.set_printoptions(precision=2)

# A = X @ X.T => A = V @ S @ V.T |
# B = X.T @ X => B = U @ S @ U.T |
#                                | => X = U @ S @ V.T where (U, S), (V, S) - spectral decomposition of A and B

x_dim = 30
samples_dim = 60

A = torch.empty(samples_dim, samples_dim)
for i in range(samples_dim):
    for j in range(samples_dim):
        A[i, j] = exp(-abs(i - j)/100)

B = torch.eye(x_dim, x_dim)

eig_b, V_b = torch.linalg.eig(B)
eig_b = eig_b.real
V_b = V_b.real
print(eig_b)
print(V_b)


def eig_to_sigma(eig):
    sigma = torch.diag(torch.sqrt(eig))
    sigma = torch.concat([sigma, torch.zeros(samples_dim - x_dim, x_dim)], dim=0)
    return sigma


S_a = S_a @ S_a.timeline
print(eig_b)
print(S_a)


def residual(V):
    subject = torch.norm(V @ S_a @ V.timeline - B)
    cond = torch.norm(V @ V.timeline - torch.eye(samples_dim))
    return subject + cond


def optimize():
    V = torch.eye(samples_dim, requires_grad=True)
    iters = 1000

    opt = optim.SGD([V], 0.01)
    for _ in range(iters):
        loss = residual(V)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(loss)



optimize()
