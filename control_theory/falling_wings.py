from typing import Dict

import control as ct
import numpy as np
from matplotlib import pyplot as plt


def falling_wings_observe(_: float, x: np.ndarray, *args, **kwargs):
    c_mat = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    return c_mat @ x


def falling_wings_update(_: float, x: np.ndarray, u: np.ndarray, params: Dict):
    z_vel_square = x[2]**2

    x_dot = np.zeros_like(x)
    const_term = 0.5 * params['rho_air'] * params['area'] / params['mass']
    x_dot[0] = z_vel_square * np.sin(u[0]) * const_term * params['C_alpha'] + params['wind'][0] / params['mass']
    x_dot[1] = z_vel_square * np.sin(u[1]) * const_term * params['C_beta'] + params['wind'][1] / params['mass']
    x_dot[2] = params['g'] + z_vel_square * const_term * params['C_drag'] + params['wind'][2] / params['mass']

    x_dot[3] = x[0]
    x_dot[4] = x[1]
    x_dot[5] = x[2]
    return x_dot


def targeting_system_update(t: float, x: np.ndarray, u: np.ndarray, params: Dict):
    x_dot = falling_wings_update(t, x, u, params)
    x_dot_targeting = np.array([x_dot[3], x_dot[4]])
    return x_dot_targeting


def main():
    params = {
        'rho_air': 1.293,
        'C_alpha': 0.1,
        'C_beta': 0.1,
        'C_drag': 0.3,
        'area': 0.02,
        'mass': 1.2,
        'g': -9.81,
        'wind': np.array([0.0, 0.0, 0.0]),
    }

    falling_wing_sys = ct.NonlinearIOSystem(
        updfcn=falling_wings_update,
        outfcn=falling_wings_observe,
        inputs=('u_alpha', 'u_beta'),
        states=('x velocity', 'y velocity', 'z velocity', 'x position', 'y position', 'z position'),
        params=params
    )

    u = np.array([15, 15])
    u = np.deg2rad(u)
    x0 = np.array([0.0, 0.0, 0.0, 20.0, 20.0, 500.0])
    x = x0.copy()
    x_linear = x0.copy()
    dt = 0.05
    t = 0

    trj = []
    trj_linear = []
    while x[-1] >= 0.0:
        linear_sys = falling_wing_sys.linearize(x0=x, u0=u, t=t)
        dxdt = falling_wing_sys.dynamics(t=dt, x=x, u=u)
        dxdt_linear = linear_sys.dynamics(t=dt, x=x, u=u)
        Q = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0001, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0001],
        ])
        R = np.eye(linear_sys.ninputs)
        try:
            # print(ct.ctrb(linear_sys.A, linear_sys.B))
            gain_K, S, E = ct.dlqr(linear_sys.A, linear_sys.B, Q, R)
            print(gain_K)
            u = -gain_K @ x
            print(u)
        except np.linalg.LinAlgError as e:
            print(e)
        t += dt
        x += dxdt * dt
        x_linear += dxdt_linear * dt

        trj.append(np.copy(x))
        trj_linear.append(np.copy(x_linear))

    trj = np.array(trj).T
    trj_linear = np.array(trj_linear).T
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot(trj[3], trj[4], trj[5], label='nl')
    ax.plot(trj_linear[3], trj_linear[4], trj_linear[5], label='linear')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
