import math

import torch.utils.data
from torch import distributions


def oscillation(time, ampl, freq, decay, noise_dist: distributions.Distribution):
    return ampl * torch.sin(time * freq) * torch.exp(-time * decay) \
           + noise_dist.sample(time.shape)


class OscillationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 n_samples, time_domain, c_ticks,
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
        c_count = c_ticks.size(0)
        time_domain = time_domain.view(1, seq_length).repeat(n_samples, 1)

        amplitudes = ampl_dist.sample([n_samples, 1]).repeat(1, seq_length)
        freqs = freq_dist.sample([n_samples, 1]).repeat(1, seq_length)
        decays = decay_dist.sample([n_samples, 1]).repeat(1, seq_length)

        noise_dist = distributions.Normal(noise_mean, noise_sigma)
        self.samples = torch.empty(n_samples, c_count, seq_length)
        for i in range(c_count):
            oscillations = oscillation(time_domain, amplitudes, freqs, decays, noise_dist) + c_ticks[i]
            self.samples[:, i, :] = oscillations

        self.sep = int(xy_separation * seq_length)
        self.c_count = c_count

    def __len__(self):
        return self.samples.size(0)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        x = torch.transpose(self.samples[index, :, :self.sep], 0, 1)
        y = torch.transpose(self.samples[index, :, self.sep:], 0, 1)
        return x, y

    @property
    def x_shape(self):
        return torch.Size((self.sep, self.c_count))

    @property
    def y_shape(self):
        return torch.Size((self.samples.size(2) - self.sep, self.c_count))

    @classmethod
    def default(cls, size, seq_len):
        time = torch.linspace(0, math.pi, seq_len)
        initial_conditions = torch.linspace(0, math.pi/6, 12)
        xy_sep = 0.8
        dataset = cls(size, time, initial_conditions, xy_sep,
                      1.0, 0.01,
                      30.0, 5.0,
                      1.0, 0.3,
                      0.0, 0.001)
        return dataset, time, initial_conditions


if __name__ == '__main__':
    def test():
        import matplotlib.pyplot as plt
        dataset_size = 10
        seq_len = 400
        xy_sep = 0.8
        dataset, time, _ = OscillationDataset.default(dataset_size, seq_len)

        sep = int(seq_len * xy_sep)
        for i in range(dataset_size):
            x, y = dataset[i]
            for ph in range(dataset.c_count):
                plt.plot(time[:sep], x[:, ph])
                plt.plot(time[sep:], y[:, ph])
            plt.show()
    test()
