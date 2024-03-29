import matplotlib.pyplot as plt
import numpy as np


class RampTrj:
    def __init__(self, x0, vel0, vel1, vel_max, acc, t_dur):
        self.x0 = x0
        self.vel0 = vel0
        self.vel1 = vel1
        self.vel_max = vel_max
        self.acc = acc
        self.t_dur = t_dur
        self.t0 = (vel_max - vel0) / acc
        self.t1 = t_dur - (vel_max - vel1) / acc  # todo change vel_max if t0 > t1

    def at(self, t):
        x = self.x0 \
                + self.vel0 * self.t0 + 0.5 * self.acc * self.t0 ** 2 \
                + self.vel_max * (self.t1 - self.t0) \
                + self.vel_max * (self.t_dur - self.t1) \
                - 0.5 * self.acc * (self.t_dur - self.t1) ** 2
        if t < self.t0:
            x = self.x0 + self.vel0 * t + 0.5 * self.acc * t ** 2
        elif t < self.t1:
            x = self.x0 \
                + self.vel0 * self.t0 + 0.5 * self.acc * self.t0 ** 2 \
                + self.vel_max * (t - self.t0)
        elif t < self.t_dur:
            x = self.x0 \
                + self.vel0 * self.t0 + 0.5 * self.acc * self.t0 ** 2 \
                + self.vel_max * (self.t1 - self.t0) \
                + self.vel_max * (t - self.t1) \
                - 0.5 * self.acc * (t - self.t1) ** 2
        return x

    def sample(self, time: np.ndarray):
        xs = []
        for t in time:
            xs.append(self.at(t))
        return np.array(xs)


def ramp_time(x0, x1, vel0, vel1, vel_max, acc):
    delta_t0 = (vel_max - vel0) / acc
    delta_t1 = (vel_max - vel1) / acc
    delta_t_plato = 15

    def f(t):
        return vel0 * delta_t0 + 0.5 * acc * delta_t0 ** 2 + delta_t_plato * vel_max - vel1 * delta_t1 + 0.5 * acc * delta_t1 ** 2


def main():
    x0 = 13.0
    vel0 = 0.0
    vel1 = 0.0
    vel_max = 15.0
    acc = 2.0
    t_dur = 11.0
    trj = RampTrj(x0, vel0, vel1, vel_max, acc, t_dur)
    time = np.linspace(0.0, t_dur + 10, 250)
    plt.plot(time, trj.sample(time))
    plt.show()


if __name__ == '__main__':
    main()
