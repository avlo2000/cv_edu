import matplotlib.pyplot as plt
import numpy as np


def main():
    n_samples = 50_000
    time = np.linspace(0.0, 100.0, n_samples)
    x = np.cos(time)
    measurement_noise = np.random.normal(0.0, scale=0.1, size=n_samples)
    x_n = x + measurement_noise

    plt.subplot(311)
    plt.plot(time, x_n)
    plt.plot(time, x)

    x_n_int = np.cumsum(x_n * time)
    x_int = np.cumsum(x * time)

    plt.subplot(312)
    plt.plot(time, x_n_int)
    plt.plot(time, x_int)

    x_n_int_int = np.cumsum(x_n_int * time)
    x_int_int = np.cumsum(x_int * time)

    plt.subplot(313)
    plt.plot(time, x_n_int_int)
    plt.plot(time, x_int_int)

    plt.show()

    measurement_noise = np.random.normal(0.0, scale=0.1, size=n_samples)
    plt.hist(measurement_noise, bins=1000)
    noise_int = np.cumsum(measurement_noise * time)
    print(noise_int.mean())
    plt.hist(noise_int, bins=1000)
    plt.show()


if __name__ == '__main__':
    main()
