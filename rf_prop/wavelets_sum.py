import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import wavelets


n_w = 500
n = 4000
wavelet = wavelets.ricker(n_w, 20)
plt.plot(wavelet)
plt.show()

space = np.zeros(n)
shifts = np.random.randint(low=0, high=n - n_w, size=10000)
for shift in shifts:
    space[shift:shift + n_w] += np.real(wavelet)

plt.plot(space)
plt.show()
