import math

import torch.utils.data
from torch import distributions


def oscillation(time, ampl, freq, decay, noise_dist: distributions.Distribution):
    return ampl * torch.sin(time * freq) * torch.exp(-time * decay) + noise_dist.sample(time.shape)


class OscillationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 size, time_domain, phase_domain,
                 xy_separation,
                 ampl_mean, ampl_sigma,
                 freq_mean, freq_sigma,
                 decay_mean, decay_sigma,
                 noise_mean, noise_sigma,
                 ):
        assert 0.0 < xy_separation < 1.0 and "xy_separation must be in range (0.0, 1.0)"

        ampl_dist = distributions.Normal(ampl_mean, ampl_sigma)
        freq_dist = distributions.Normal(freq_mean, freq_sigma)
        decay_dist = distributions.Normal(decay_mean, decay_sigma)

        seq_length = time_domain.size(0)
        time_domain = time_domain.view(1, seq_length).repeat(size, 1)

        amplitudes = ampl_dist.sample([size, 1]).repeat(1, seq_length)
        freqs = freq_dist.sample([size, 1]).repeat(1, seq_length)
        decays = decay_dist.sample([size, 1]).repeat(1, seq_length)

        noise_dist = distributions.Normal(noise_mean, noise_sigma)
        self.samples = oscillation(time_domain, amplitudes, freqs, decays, noise_dist)
        self.sep = int(xy_separation * seq_length)

    def __len__(self):
        return self.samples.size(0)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        return self.samples[index, :self.sep], self.samples[index, self.sep:]

    def x_size(self):
        return self.sep

    def y_size(self):
        return self.samples.size(1) - self.sep

    @classmethod
    def default(cls, size, seq_len):
        time = torch.linspace(0, math.pi, seq_len)
        xy_sep = 0.8
        dataset = cls(size, time, xy_sep,
                      1.0, 0.01,
                      30.0, 5.0,
                      1.0, 0.3,
                      0.0, 0.001)
        return dataset


if __name__ == '__main__':
    def test():
        import matplotlib.pyplot as plt
        dataset_size = 10
        seq_len = 400
        time = torch.linspace(0, math.pi, seq_len)
        xy_sep = 0.8
        dataset = OscillationDataset.default(dataset_size, seq_len)

        sep = int(seq_len * xy_sep)
        for i in range(dataset_size):
            x, y = dataset[i]
            plt.plot(time[:sep], x)
            plt.plot(time[sep:], y)
            plt.show()


    test()
