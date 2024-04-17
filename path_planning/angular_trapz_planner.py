from typing import List, Union

import numpy as np
import velocity_trapz


def angle_diff(a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    diff = a - b
    while diff < -np.pi:
        diff += 2 * np.pi
    while diff > np.pi:
        diff -= 2 * np.pi
    return diff


def plan_angular(poses: List[float], max_vel: float, acc: float):
    def plan_rec(idx: int, vel: float):
        if idx + 1 == len(poses):
            return 0.0
        vels_next = [0.0]
        if vel != 0:
            vels_next.append(np.sign(vel) * max_vel)
        else:
            vels_next.append(+max_vel)
            vels_next.append(-max_vel)

        durations = []
        for vel_next in vels_next:
            dur = velocity_trapz.trapz_time(abs(angle_diff(poses[idx], poses[idx + 1])), vel, vel_next, max_vel, acc)
            durations.append(dur + plan_rec(idx + 1, vel_next))
            if idx + 2 == len(poses):
                print(durations[-1])

        return min(*durations)

    t = plan_rec(0, 0.0)
    print(t)


if __name__ == '__main__':
    def main():
        poses = [0.0, np.pi, np.pi / 2, 1.5 * np.pi, 0.0]
        max_vel = 0.5  # rad/s
        acc = 0.2  # rad/s^2

        plan_angular(poses, max_vel, acc)

    main()
