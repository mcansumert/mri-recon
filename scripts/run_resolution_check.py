"""Classical resolution check.

Thick bars sit inside the acquired k-space. Thin bars sit outside it.
Zeroing the missing frequencies removes the thin bars and leaves no energy
in the frequencies that were never acquired.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri_recon.fftc import fft2c, ifft2c


def resolution_phantom(size: int = 128) -> torch.Tensor:
    yy = torch.arange(size).view(-1, 1)
    image = torch.full((size, size), 0.3)
    thick = ((yy // 8) % 2).float()
    thin = ((yy // 2) % 2).float()
    image[8:40] = 0.3 + 0.7 * thick[8:40]
    image[88:120] = 0.3 + 0.7 * thin[88:120]
    return image.to(torch.complex64)


def lowpass(image: torch.Tensor, kept: int) -> tuple[torch.Tensor, torch.Tensor]:
    spectrum = fft2c(image)
    num_lines = spectrum.shape[-2]
    mask = torch.zeros(num_lines, dtype=spectrum.real.dtype)
    start = (num_lines - kept) // 2
    mask[start : start + kept] = 1
    recon = ifft2c(spectrum * mask.view(-1, 1))
    return recon, mask


def region_std(image: torch.Tensor, y0: int, y1: int) -> float:
    return float(image.real[y0:y1].std())


def main() -> None:
    image = resolution_phantom()
    recon, mask = lowpass(image, kept=32)
    thick = region_std(recon, 8, 40)
    thin = region_std(recon, 88, 120)
    spectrum = fft2c(recon)
    outside = (1.0 - mask).view(-1, 1)
    outside_energy = torch.sum(spectrum.abs() ** 2 * outside)
    total_energy = torch.sum(spectrum.abs() ** 2).clamp_min(1e-12)
    fraction = float((outside_energy / total_energy).detach())
    print(f"thick_std {thick:.4f}")
    print(f"thin_std {thin:.4f}")
    print(f"outside_energy {fraction:.3e}")
    if thick <= thin * 3:
        raise SystemExit("acquired bars are not distinguishable from deleted bars")
    if fraction > 1e-5:
        raise SystemExit("reconstruction invented frequencies outside the acquisition")
    print("resolution_check_ok")


if __name__ == "__main__":
    main()