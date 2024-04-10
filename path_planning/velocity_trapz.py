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
        self.t0 = abs(vel_max - vel0) / acc
        self.t1 = t_dur - abs(vel_max - vel1) / acc
        if self.t0 > self.t1:
            self.vel_max = 0.5 * (acc * t_dur + vel0 + vel1)
            print(f"self.vel_max: {self.vel_max}")

    def at(self, t):
        if self.t1 > self.t0:
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
        else:
            t0 = abs(self.vel_max - self.vel0) / self.acc
            t1 = abs(self.vel_max - self.vel1) / self.acc
            x = self.x0 \
                + self.vel0 * t0 + 0.5 * self.acc * t0 ** 2 \
                + self.vel_max * (t1 - t0) \
                - 0.5 * self.acc * (t1 - t0) ** 2
            if t < t0:
                x = self.x0 \
                    + self.vel0 * t + 0.5 * self.acc * t ** 2
            elif t < t1:
                x = self.x0 \
                    + self.vel0 * t0 + 0.5 * self.acc * t0 ** 2 \
                    + self.vel_max * (t - t0) \
                    - 0.5 * self.acc * (t - t0) ** 2
            return x

    def last_x(self):
        return self.at(self.t_dur)

    def sample(self, time: np.ndarray):
        xs = []
        for t in time:
            xs.append(self.at(t))
        return np.array(xs)


def trapz_time(dist, vel0, vel1, vel_max, acc):
    vel_max_hat = np.sqrt(acc * dist + 0.5 * vel0 ** 2 + 0.5 * vel1 ** 2)
    print(f"vel_max_hat: {vel_max_hat}")
    if vel_max > vel_max_hat:
        dt0 = abs(vel_max_hat - vel0) / acc
        dt1 = abs(vel_max_hat - vel1) / acc
        return dt0 + dt1

    dt0 = (vel_max - vel0) / acc
    dt1 = (vel_max - vel1) / acc
    dx0 = vel0 * dt0 + 0.5 * acc * dt0 ** 2
    dx1 = vel_max * dt1 - 0.5 * acc * dt1 ** 2
    t1 = (dist + vel_max * dt0 - dx0 - dx1) / vel_max
    return t1 + dt1


def main():
    x0 = 13.0
    vel0 = 1.0
    vel1 = 4.0
    vel_max = 15.0
    acc = 2.0
    t_dur = 4 
    trj = RampTrj(x0, vel0, vel1, vel_max, acc, t_dur)
    time = np.linspace(0.0, t_dur, 250)
    print(trapz_time(x0, trj.last_x(), vel0, vel1, vel_max, acc))
    plt.plot(time, trj.sample(time))
    plt.show()


if __name__ == '__main__':
    main()
