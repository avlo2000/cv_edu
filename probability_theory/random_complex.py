import matplotlib.pyplot as plt
import numpy as np


count = 500_000
angle = np.random.uniform(0, np.pi, size=count)
# angle = np.random.normal(0, np.pi, size=count)
c = np.exp(1.0j * angle)
hist_x, hist_y = np.histogram(np.imag(-c), bins=500)

plt.subplot(211)
plt.plot(hist_x, hist_y[:-1])


count = 500_000
angle = np.random.uniform(0, np.pi, size=count)
# angle = np.random.normal(0, np.pi, size=count)
c = np.exp(angle)
hist_x, hist_y = np.histogram(c, bins=500)

plt.subplot(212)
plt.plot(hist_x, hist_y[:-1])
plt.show()
