import torch
import matplotlib.pyplot as plt

mean1 = torch.scalar_tensor(1.0)
sigma1 = torch.scalar_tensor(1.0)
normal1 = torch.distributions.Normal(mean1, sigma1)

mean2 = torch.scalar_tensor(2.0)
sigma2 = torch.scalar_tensor(3.0)
normal2 = torch.distributions.Normal(mean2, sigma2)

kl_div = torch.log(sigma2/sigma1) + (sigma1**2 + (mean1 - mean2)**2) / (2*sigma2**2) - 0.5
print(kl_div)


space = torch.linspace(-10, 10, 500)

probs1 = torch.exp(normal1.log_prob(space))
probs2 = torch.exp(normal2.log_prob(space))

kl_div = torch.trapz((probs1 * torch.log(probs1/probs2)), space)

print(kl_div)

plt.plot(space, probs1)
plt.plot(space, probs2)
plt.show()
