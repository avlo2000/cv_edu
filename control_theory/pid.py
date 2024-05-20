import numpy as np


class PIDControl:
    def __init__(self):
        self.p = 1.8
        self.i = 0.00
        self.d = 0.05
        self._prev_err = 0.0
        self._sum_error = 0.0

    def control(self, state: np.ndarray) -> np.ndarray:
        pose = state[:2]
        trg_pose = state[3:5]
        err = trg_pose - pose
        cmd = self.p * err + self.d * (err - self._prev_err) + self.i * self._sum_error
        self._prev_err = err
        self._sum_error += err
        return cmd
