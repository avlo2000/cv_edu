import numpy as np
import torch
import cv2
import time

from rf_prop.emsim import EMSim
from rf_prop.sources import PointSource
from rf_prop.reflectors import SegmentReflector

print(torch.cuda.is_available())

wave_speed = 2.1
freq = torch.tensor(0.2)
wavelen = wave_speed / freq

em_sim = EMSim(
    -50, 50, 600,
    -50, 50, 600
)

for i in range(5):
    em_sim.add_point_source(PointSource(torch.tensor([0.0, i * wavelen ])))
mx_amp = 20
dists, mask = em_sim.calculate_distmaps()


def wave_front(p: torch.Tensor, dists: torch.Tensor, t: float) -> torch.Tensor:
    m = dists > t * wave_speed
    for i in range(dists.shape[0]):
        p[m[i, :, :]] = 0.0
    return p


def potential(t: float) -> torch.Tensor:
    w = 2.0 * torch.pi * freq
    k = w / wave_speed
    amplitudes = mask * mx_amp * torch.sin(k * dists - w * t) / dists ** 2
    p = torch.sum(amplitudes, dim=0)
    return p


def show_potential(p: torch.Tensor):
    print(f"MAX: {torch.max(p).item()}, MIN: {torch.min(p).item()}")
    p = (p + mx_amp) / 2 * mx_amp
    print()
    p_np = p.numpy().squeeze() * 255.0
    im_color = cv2.applyColorMap(p_np.astype(np.uint8), cv2.COLORMAP_COOL)
    cv2.imshow("waves", im_color)
    cv2.waitKey()


def main():
    for t in torch.linspace(0.0, 600, 900):
        t0 = time.time()
        p = potential(t)
        print(time.time() - t0)
        show_potential(p)


if __name__ == '__main__':
    main()
