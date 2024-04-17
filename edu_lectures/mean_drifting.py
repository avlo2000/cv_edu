import matplotlib.pyplot as plt
import numpy as np


n_samples = np.linspace(start=500, stop=100_000, num=50)
means = np.empty_like(n_samples)
for i, n in enumerate(n_samples):
    rnd_n = 500
    data = np.random.normal(0.0, scale=10.0, size=(int(n), rnd_n))
    means[i] = np.mean(data)

plt.plot(n_samples, means)
plt.show()
