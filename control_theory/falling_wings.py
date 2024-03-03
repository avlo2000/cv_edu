from typing import Dict

import control
import numpy as np
from matplotlib import pyplot as plt


def update_fn(_: float, x: np.ndarray, u: np.ndarray, params: Dict):
    z_vel_square = x[2]**2

    x_dot = np.zeros_like(x)
    const_term = 0.5 * params['rho_air'] * params['area'] / params['mass']
    x_dot[0] = z_vel_square * np.sin(u[0]) * const_term * params['C_alpha']
    x_dot[1] = z_vel_square * np.sin(u[1]) * const_term * params['C_beta']
    x_dot[2] = params['g'] + z_vel_square * const_term * params['C_drag']

    x_dot[3] = x[0]
    x_dot[4] = x[1]
    x_dot[5] = x[2]
    return x_dot


def create_transfer_functions(params: Dict):
    const_term = 0.5 * params['rho_air'] * params['area'] / params['mass']
    const_term * params['C_alpha']
    const_term * params['C_beta']
    control.tf()


def main():
    io_sys = control.NonlinearIOSystem(
        update_fn,
        inputs=('u_alpha', 'u_beta'),
        states=('x velocity', 'y velocity', 'z velocity', 'x position', 'y position', 'z position'),
        params={
            'rho_air': 1.293,
            'C_alpha': 1.2,
            'C_beta': 1.2,
            'C_drag': 0.3,
            'area': 0.4,
            'mass': 1.5,
            'g': -9.81
        }
    )

    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 500.0])
    u = np.array([[15.0, -15.0]]).T.repeat(500, axis=1)
    u = np.deg2rad(u)

    timeline = np.linspace(0, 10, 500)

    t, y = control.input_output_response(io_sys, timeline, u, x0)
    control.ss2tf()

    plt.subplot(211)
    plt.plot(t, y[0], c='red')
    plt.plot(t, y[1], c='green')
    plt.plot(t, y[2], c='blue')
    plt.subplot(212)
    plt.plot(t, y[3], c='red')
    plt.plot(t, y[4], c='green')
    plt.plot(t, y[5], c='blue')
    plt.show()


if __name__ == '__main__':
    main()
