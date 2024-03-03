import torch
import torch.optim as optim
torch.set_printoptions(precision=2)

# A = X @ X.T => A = V @ S @ V.T |
# B = X.T @ X => B = U @ S @ U.T |
#                                | => X = U @ S @ V.T where (U, S), (V, S) - spectral decomposition of A and B

x_dim = 3
samples_dim = 6

A = torch.rand(x_dim, x_dim)
A = A @ A.T

B = torch.rand(samples_dim, samples_dim)
B = B @ B.T

eig_a, U_a = torch.linalg.eig(A)
eig_a = eig_a.real
U_a = U_a.real
print(eig_a)
print(U_a)


def eig_to_sigma(eig):
    sigma = torch.diag(torch.sqrt(eig))
    sigma = torch.concat([sigma, torch.zeros(samples_dim - x_dim, x_dim)], dim=1)
    return sigma


S_a = eig_to_sigma(eig_a)
print(S_a)


def residual(V):
    subject = torch.norm(V @ S_a.T @ S_a @ V.timeline - B)
    cond = torch.norm(V @ V.timeline - torch.eye(samples_dim))
    return subject + 5 * cond


def optimize():
    V = torch.eye(samples_dim, requires_grad=True)
    iters = 100

    opt = optim.Adam([V], 0.05)
    for _ in range(iters):
        loss = residual(V)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(loss)
    print(V)
    print(V @ V.T)


optimize()
