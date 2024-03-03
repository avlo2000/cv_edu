import numpy as np

from path_planning.dubins2d import DubinsPath2d, Waypoint2d, calc_dubins_path


class Waypoint3d:
    def __init__(self, x: float, y: float, z: float, psi: float):
        self.x = x
        self.y = y
        self.z = z
        self.psi = psi

    def to2d(self):
        return Waypoint2d(self.x, self.y, self.psi)


class DubinsPath3d:
    def __init__(self, projection: DubinsPath2d, z0: float, z1: float):
        self.projection = projection
        self.z0 = z0
        self.z1 = z1

    def trace(self, n_points) -> np.ndarray:
        trace2d = self.projection.trace(n_points)
        elevations = np.linspace(self.z0, self.z1, n_points)
        return np.concatenate((
            trace2d[:, :2],
            elevations[:, np.newaxis],
            trace2d[:, 2][:, np.newaxis]
        ), axis=1)


def calc_dubins_path_3d(
        wp0: Waypoint3d,
        wp1: Waypoint3d,
        min_r: float,
        max_tg: float,
        binsearch_iters: int
) -> DubinsPath3d:
    lower_r_bound = float(min_r)
    upper_r_bound = 100_000.0  # 10 km, hope enough :)
    elevation = abs(wp1.z - wp0.z)

    mid_r = 0.5 * (upper_r_bound + lower_r_bound)
    path2d_mid = calc_dubins_path(wp0.to2d(), wp1.to2d(), mid_r)
    for _ in range(binsearch_iters - 1):
        if elevation > path2d_mid.length * max_tg:
            lower_r_bound = mid_r
        else:
            upper_r_bound = min_r

        mid_r = 0.5 * (upper_r_bound + lower_r_bound)
        path2d_mid = calc_dubins_path(wp0.to2d(), wp1.to2d(), mid_r)
    return DubinsPath3d(path2d_mid, wp0.z, wp1.z)


def main():
    wp0 = Waypoint3d(0, 78, 100, 0)
    wp1 = Waypoint3d(0, 0, 511, 180)

    path3d = calc_dubins_path_3d(wp0, wp1, min_r=200, max_tg=0.1, binsearch_iters=20)
    trace = path3d.trace(200)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(trace[:, 0], trace[:, 1], trace[:, 2], marker='o')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Path')

    plt.show()


if __name__ == '__main__':
    main()
