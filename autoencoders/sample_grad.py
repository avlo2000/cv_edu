import torch

sigma = torch.tensor(3.0, requires_grad=True)
mean = torch.tensor(1.0, requires_grad=True)
normal = torch.distributions.Normal(0.0, 1.0)
point = mean + sigma * normal.sample([1])
print(point)
point.backward()
print(sigma.grad)
