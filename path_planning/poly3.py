from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent


class Waypoint2d:

    def __init__(self, x: float, y: float, psi: float):
        self.x = x
        self.y = y
        self.psi = psi

    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y) + ", psi: " + str(self.psi)


class Poly5Path:
    def __init__(self, params: np.ndarray, t0: float, t1: float):
        self.params = params
        self.t0 = t0
        self.t1 = t1

    def trace(self, n_points: int):
        t = np.linspace(self.t0, self.t1, n_points)
        return t, self.substitute(t)

    def substitute(self, t: np.ndarray) -> np.ndarray:
        return self.params[0] * t ** 3 + \
               self.params[1] * t ** 2 + \
               self.params[2] * t + \
               self.params[3]


def calc_path(wp0: Waypoint2d, wp1: Waypoint2d) -> Poly5Path:
    der0 = np.tan(wp0.psi)
    der1 = np.tan(wp1.psi)

    mat = np.array([
        [wp0.x**3, wp0.x**2, wp0.x, 1],
        [wp1.x**3, wp1.x**2, wp1.x, 1],
        [3*wp0.x**2, 2*wp0.x, 1, 0],
        [3*wp1.x**2, 2*wp1.x, 1, 0],
    ])
    b = np.array([wp0.y, wp1.y, der0, der1])
    params = np.linalg.pinv(mat) @ b
    return Poly5Path(params, wp0.x, wp1.x)


def main():
    wp0 = Waypoint2d(0, 0, np.pi)

    ax = plt.gca()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    def on_click(event: MouseEvent) -> Any:
        wp1 = Waypoint2d(event.xdata, event.ydata, 1.001*np.pi/2)
        path = calc_path(wp0, wp1)

        x, y = path.trace(200)

        plt.arrow(wp0.x, wp0.y, 0.2*np.cos(wp0.psi), 0.2*np.sin(wp0.psi))
        plt.arrow(wp1.x, wp1.y, 0.2*np.cos(wp1.psi), 0.2*np.sin(wp1.psi))
        plt.plot(x, y)
        plt.draw()

    plt.connect('button_press_event', on_click)

    plt.show()
    plt.draw()


if __name__ == '__main__':
    main()
