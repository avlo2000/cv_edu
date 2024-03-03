import torch


x_dim = 5
samples_dim = 50

A = torch.rand(x_dim, x_dim)
A = A @ A.T

B = torch.rand(samples_dim, samples_dim)
B = B @ B.T

eig_a, V_a = torch.linalg.eig(A)
eig_b, V_b = torch.linalg.eig(A)

S_a = torch.diag(torch.sqrt(eig_a))
S_b = torch.diag(torch.sqrt(eig_b))

X = (V_a @ S_a @ V_b.timeline @ S_b).real
print(B)
print(X @ X.timeline)
