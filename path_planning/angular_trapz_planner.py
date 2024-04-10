from typing import List

import numpy as np
import velocity_trapz


class State:
    def __init__(self, pose: float, vel: float):
        self.pose = pose
        self.vel = vel


def plan(poses: List[float], max_vel: float, acc: float):
    def plan_rec(idx: int, vel0: float, vel1: float):
        dur = velocity_trapz.trapz_time(
            poses[idx], poses[idx + 1], vel0, vel1, max_vel, acc
        )
