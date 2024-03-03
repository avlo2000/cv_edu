import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb


def binary_likelihood(data: np.ndarray, hypothesis: float):
    ones = np.sum(data)
    zeros = len(data) - ones
    prob = comb(len(data), ones, exact=False) * (hypothesis ** ones) * (1.0 - hypothesis) ** zeros
    return prob


def main():
    real_model = 0.6
    data = (np.random.uniform(0.0, 1.0, size=600) < real_model)
    binary_likelihood(data, real_model)
    hypothesises = np.linspace(0.0, 1.0, 4000)
    likelihood = np.zeros_like(hypothesises)
    for i, hypothesis in enumerate(hypothesises):
        likelihood[i] = binary_likelihood(data, hypothesis)

    print(np.trapz(likelihood, hypothesises))
    plt.plot(hypothesises, likelihood)
    plt.show()


if __name__ == '__main__':
    main()
