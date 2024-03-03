import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from matplotlib import cm
from matplotlib.ticker import LinearLocator


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


def norm_cdf(x, mu, sigma):
    t = x - mu
    y = 0.5 * special.erf(-t / (sigma * np.sqrt(2.0)))
    y[y > 1.0] = 1.0
    return y


def normal_likelihood(data, mean, std):
    eps = 0.001
    lb = data - eps
    ub = data + eps
    pr_lb = norm_cdf(lb, mean, std)
    pr_ub = norm_cdf(ub, mean, std)
    return np.sum(pr_ub - pr_lb)


def main():
    real_model_mean = 0.0
    real_model_std = 0.1
    data = np.random.normal(real_model_mean, real_model_std, size=100)

    hypothesis_mean = np.linspace(-2.0, 2.0, 60)
    hypothesis_std = np.linspace(0, 2.0, 60)
    likelihood = np.zeros([hypothesis_mean.shape[0], hypothesis_mean.shape[0]])
    for i, mean in enumerate(hypothesis_mean):
        for j, std in enumerate(hypothesis_mean):
            likelihood[i, j] = normal_likelihood(data, mean, std)

    i_max = np.argmax(np.max(likelihood, axis=1))
    j_max = np.argmax(np.max(likelihood, axis=0))
    print(f"Most likely mean = {hypothesis_mean[j_max]} and std = {hypothesis_std[i_max]}")
    mesh_mean, mesh_std = np.meshgrid(hypothesis_mean, hypothesis_std)
    surf = ax.plot_surface(mesh_mean, mesh_std, likelihood, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    main()
