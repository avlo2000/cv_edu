import torch

sigma = torch.tensor(1.0, requires_grad=True)
mean = torch.tensor(0.0, requires_grad=True)
normal = torch.distributions.Normal(mean, sigma)
point = normal.sample([1])
print(point)
point.backward()
print(sigma.grad)
