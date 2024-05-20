import numpy as np


class Quantizer:
    def __init__(self, min_vals: np.ndarray, max_vals: np.ndarray, num_bins: int):
        assert len(min_vals) == len(max_vals)
        self._dim = len(min_vals)
        self._bins = np.empty([self._dim, num_bins])
        self._max_vals = max_vals
        self._min_vals = min_vals
        for i, (mn, mx) in enumerate(zip(min_vals, max_vals)):
            self._bins[i] = np.linspace(mn, mx, num=num_bins)

    def quantize(self, vals: np.ndarray) -> np.ndarray:
        idx = np.empty_like(vals, dtype=int)
        for i, v in enumerate(vals):
            idx[i] = np.digitize(v, self._bins[i])
        return idx - 1

    def dequant(self, idx: np.ndarray) -> np.ndarray:
        return self._min_vals + (self._max_vals - self._min_vals) * (idx + 1) / self.num_bins

    @property
    def num_bins(self):
        return self._bins.shape[1]

    @property
    def dim(self):
        return self._dim


if __name__ == '__main__':
    def main():
        q = Quantizer(np.array([0.0, -5.0, 0.0]), np.array([1.0, +5.0, 0.1]), 10)
        in_data = [
            np.array([1.0, 5.0, 0.1]),
            np.array([1.0, -5.0, 0.05]),
            np.array([0.0, -5.0, 0.05]),
            np.array([0.5, 0.0, 0.02])
        ]
        for data in in_data:
            q_data = q.quantize(data)
            print(q_data)
            print(q.dequant(q_data))
            print()
    main()
