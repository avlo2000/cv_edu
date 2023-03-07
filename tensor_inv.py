import torch

a = torch.arange(60.).reshape(3, 4, 5)
b = torch.arange(4*3*7, dtype=torch.float).reshape(4, 3, 7)
print(torch.tensordot(a, b, dims=([1, 0], [0, 1])))
print(torch.tensordot(a, b, dims=([0, 1], [1, 0])))
