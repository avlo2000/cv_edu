import matplotlib.pyplot as plt
import math
import numpy as np
from enum import Enum

from dubins_planning.utils import heading_to_standard


class _SegmentType(Enum):
    L = 1
    S = 2
    R = 3


__WAY_TYPE = (
    (_SegmentType.L, _SegmentType.S, _SegmentType.L),
    (_SegmentType.L, _SegmentType.S, _SegmentType.R),
    (_SegmentType.R, _SegmentType.S, _SegmentType.L),
    (_SegmentType.R, _SegmentType.S, _SegmentType.R),
    (_SegmentType.R, _SegmentType.L, _SegmentType.R),
    (_SegmentType.L, _SegmentType.R, _SegmentType.L),
)


class Waypoint2d:

    def __init__(self, x, y, psi):
        self.x = x
        self.y = y
        self.psi = psi

    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y) + ", psi: " + str(self.psi)


class DubinsPath2d:
    def __init__(self, p_init: Waypoint2d, seg_final, turn_radius):
        self.p_init = p_init
        self.seg_lengths = seg_final
        self.turn_radius = turn_radius
        self.type = None

    @property
    def length(self):
        return (self.seg_lengths[0] + self.seg_lengths[1] + self.seg_lengths[2]) * self.turn_radius

    def trace(self, n_points: int) -> np.ndarray:
        length = self.length
        t_space = np.linspace(0, length, n_points)
        path = np.empty([n_points, 3])

        for i, t in enumerate(t_space):
            path[i] = self.__dubins_path_at_point(t)
        return path

    def __dubins_path_at_point(self, t):
        t_prime = t / self.turn_radius
        p_init = np.array([0, 0, heading_to_standard(self.p_init.psi) * math.pi / 180])

        param1 = self.seg_lengths[0]
        param2 = self.seg_lengths[1]
        mid_pt1 = self.__dubins_segment(param1, p_init, self.type[0])
        mid_pt2 = self.__dubins_segment(param2, mid_pt1, self.type[1])

        if t_prime < param1:
            end_pt = self.__dubins_segment(t_prime, p_init, self.type[0])
        elif t_prime < (param1 + param2):
            end_pt = self.__dubins_segment(t_prime - param1, mid_pt1, self.type[1])
        else:
            end_pt = self.__dubins_segment(t_prime - param1 - param2, mid_pt2, self.type[2])

        end_pt[0] = end_pt[0] * self.turn_radius + self.p_init.x
        end_pt[1] = end_pt[1] * self.turn_radius + self.p_init.y
        end_pt[2] = end_pt[2] % (2 * math.pi)

        return end_pt

    def __dubins_segment(self, seg_param, seg_init, seg_type):
        seg_end = np.array([0.0, 0.0, 0.0])
        if seg_type == _SegmentType.L:
            seg_end[0] = seg_init[0] + math.sin(seg_init[2] + seg_param) - math.sin(seg_init[2])
            seg_end[1] = seg_init[1] - math.cos(seg_init[2] + seg_param) + math.cos(seg_init[2])
            seg_end[2] = seg_init[2] + seg_param
        elif seg_type == _SegmentType.R:
            seg_end[0] = seg_init[0] - math.sin(seg_init[2] - seg_param) + math.sin(seg_init[2])
            seg_end[1] = seg_init[1] + math.cos(seg_init[2] - seg_param) - math.cos(seg_init[2])
            seg_end[2] = seg_init[2] - seg_param
        elif seg_type == _SegmentType.S:
            seg_end[0] = seg_init[0] + math.cos(seg_init[2]) * seg_param
            seg_end[1] = seg_init[1] + math.sin(seg_init[2]) * seg_param
            seg_end[2] = seg_init[2]

        return seg_end


def calc_dubins_path(wpt1: Waypoint2d, wpt2: Waypoint2d, r_min: float) -> DubinsPath2d:
    param = DubinsPath2d(wpt1, 0, 0)
    tz, pz, qz = np.zeros(6), np.zeros(6), np.zeros(6)
    param.seg_lengths = np.zeros(3)

    psi1 = heading_to_standard(wpt1.psi) * math.pi / 180
    psi2 = heading_to_standard(wpt2.psi) * math.pi / 180

    param.turn_radius = r_min
    dx = wpt2.x - wpt1.x
    dy = wpt2.y - wpt1.y

    dist = math.hypot(dx, dy)
    d = dist / param.turn_radius

    theta = math.atan2(dy, dx) % (2 * math.pi)
    alpha = (psi1 - theta) % (2 * math.pi)
    beta = (psi2 - theta) % (2 * math.pi)

    tz[0], pz[0], qz[0] = __dubins_lsl(alpha, beta, d)
    tz[1], pz[1], qz[1] = __dubins_lsr(alpha, beta, d)
    tz[2], pz[2], qz[2] = __dubins_rsl(alpha, beta, d)
    tz[3], pz[3], qz[3] = __dubins_rsr(alpha, beta, d)
    tz[4], pz[4], qz[4] = __dubins_rlr(alpha, beta, d)
    tz[5], pz[5], qz[5] = __dubins_lrl(alpha, beta, d)

    best_way = -1
    min_length = -1
    for i in range(6):
        if tz[i] == -1:
            continue
        length = tz[i] + pz[i] + qz[i]
        if length < min_length or min_length == -1:
            best_way = i
            min_length = length
            param.seg_lengths = np.array([tz[i], pz[i], qz[i]])

    param.type = __WAY_TYPE[best_way]
    return param


def __dubins_lsl(alpha, beta, d):
    tmp0 = d + math.sin(alpha) - math.sin(beta)
    tmp1 = math.atan2((math.cos(beta) - math.cos(alpha)), tmp0)
    p_squared = 2 + d * d - (2 * math.cos(alpha - beta)) + (2 * d * (math.sin(alpha) - math.sin(beta)))
    if p_squared < 0:
        p = -1
        q = -1
        t = -1
    else:
        t = (tmp1 - alpha) % (2 * math.pi)
        p = math.sqrt(p_squared)
        q = (beta - tmp1) % (2 * math.pi)
    return t, p, q


def __dubins_rsr(alpha, beta, d):
    tmp0 = d - math.sin(alpha) + math.sin(beta)
    tmp1 = math.atan2((math.cos(alpha) - math.cos(beta)), tmp0)
    p_squared = 2 + d * d - (2 * math.cos(alpha - beta)) + 2 * d * (math.sin(beta) - math.sin(alpha))
    if p_squared < 0:
        p = -1
        q = -1
        t = -1
    else:
        t = (alpha - tmp1) % (2 * math.pi)
        p = math.sqrt(p_squared)
        q = (-1 * beta + tmp1) % (2 * math.pi)
    return t, p, q


def __dubins_rsl(alpha, beta, d):
    tmp0 = d - math.sin(alpha) - math.sin(beta)
    p_squared = -2 + d * d + 2 * math.cos(alpha - beta) - 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_squared < 0:
        p = -1
        q = -1
        t = -1
    else:
        p = math.sqrt(p_squared)
        tmp2 = math.atan2((math.cos(alpha) + math.cos(beta)), tmp0) - math.atan2(2, p)
        t = (alpha - tmp2) % (2 * math.pi)
        q = (beta - tmp2) % (2 * math.pi)
    return t, p, q


def __dubins_lsr(alpha, beta, d):
    tmp0 = d + math.sin(alpha) + math.sin(beta)
    p_squared = -2 + d * d + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_squared < 0:
        p = -1
        q = -1
        t = -1
    else:
        p = math.sqrt(p_squared)
        tmp2 = math.atan2((-1 * math.cos(alpha) - math.cos(beta)), tmp0) - math.atan2(-2, p)
        t = (tmp2 - alpha) % (2 * math.pi)
        q = (tmp2 - beta) % (2 * math.pi)
    return t, p, q


def __dubins_rlr(alpha, beta, d):
    tmp_rlr = (6 - d * d + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) - math.sin(beta))) / 8
    if abs(tmp_rlr) > 1:
        p = -1
        q = -1
        t = -1
    else:
        p = (2 * math.pi - math.acos(tmp_rlr)) % (2 * math.pi)
        t = (alpha - math.atan2((math.cos(alpha) - math.cos(beta)), d - math.sin(alpha) + math.sin(beta)) + p / 2 % (
                2 * math.pi)) % (2 * math.pi)
        q = (alpha - beta - t + (p % (2 * math.pi))) % (2 * math.pi)

    return t, p, q


def __dubins_lrl(alpha, beta, d):
    tmp_lrl = (6 - d * d + 2 * math.cos(alpha - beta) + 2 * d * (-1 * math.sin(alpha) + math.sin(beta))) / 8
    if abs(tmp_lrl) > 1:
        p = -1
        q = -1
        t = -1
    else:
        p = (2 * math.pi - math.acos(tmp_lrl)) % (2 * math.pi)
        t = (-1 * alpha - math.atan2((math.cos(alpha) - math.cos(beta)),
                                     d + math.sin(alpha) - math.sin(beta)) + p / 2) % (2 * math.pi)
        q = ((beta % (2 * math.pi)) - alpha - t + (p % (2 * math.pi))) % (2 * math.pi)
    return t, p, q


def main():
    pt1 = Waypoint2d(0, 78, 0)
    pt2 = Waypoint2d(0, 0, 180)

    dubins_path = calc_dubins_path(pt1, pt2, 7)
    traced_path = dubins_path.trace(100)

    plt.plot(pt1.x, pt1.y, 'kx')
    plt.plot(pt2.x, pt2.y, 'kx')
    plt.plot(traced_path[:, 0], traced_path[:, 1], 'b-')

    plt.grid(True)
    plt.axis("equal")
    plt.title('Dubin\'s Curves Trajectory Generation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':
    main()
