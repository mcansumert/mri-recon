"""Numerical check on a synthetic multi-coil phantom.

Runs on CPU inside the already pulled PyTorch image. It does not read the
fastMRI archive and does not allocate the shared GPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_recon.fftc import fft2c, ifft2c
from mri_recon.masks import equispaced_mask, realized_acceleration, variable_density_mask
from mri_recon.metrics import nmse, psnr, ssim
from mri_recon.recon import cg_sense, zero_filled_rss


def phantom(size: int) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    image = torch.zeros(size, size)
    image[(yy / 0.7) ** 2 + (xx / 0.45) ** 2 <= 1] = 0.7
    image[(yy + 0.15) ** 2 + (xx - 0.15) ** 2 <= 0.12**2] = 1.0
    image[(yy - 0.2) ** 2 + (xx + 0.25) ** 2 <= 0.08**2] = 0.2
    bars = ((yy > -0.05) & (yy < 0.05) & (torch.abs(xx) < 0.75)).float()
    image = torch.maximum(image, bars)
    return image.to(torch.complex64)


def sensitivities(size: int, n_coils: int) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    maps = []
    for coil in range(n_coils):
        angle = 2.0 * math.pi * coil / n_coils
        center_y = 0.55 * math.sin(angle)
        center_x = 0.55 * math.cos(angle)
        magnitude = torch.exp(-((yy - center_y) ** 2 + (xx - center_x) ** 2) / 0.45)
        phase = torch.exp(1j * torch.tensor(0.4 * coil) * xx)
        maps.append(magnitude.to(torch.complex64) * phase.to(torch.complex64))
    sens = torch.stack(maps, dim=0)
    rss = torch.sqrt(torch.sum(sens.abs() ** 2, dim=0)).clamp_min(1e-8)
    return sens / rss


def main() -> None:
    torch.manual_seed(0)
    size = 192
    image = phantom(size)
    sens = sensitivities(size, n_coils=8)
    kspace = fft2c(sens * image.unsqueeze(0))

    roundtrip = ifft2c(fft2c(image))
    roundtrip_error = torch.max(torch.abs(roundtrip - image)).item()
    print(f"roundtrip_max_abs {roundtrip_error:.3e}")
    if roundtrip_error > 1e-5:
        raise SystemExit("FFT roundtrip failed")

    full_mask = torch.ones(size)
    full_rss = zero_filled_rss(kspace, full_mask)
    target = image.abs()
    full_nmse = nmse(full_rss, target).item()
    print(f"full_rss_nmse {full_nmse:.3e}")
    if full_nmse > 1e-5:
        raise SystemExit("fully sampled RSS does not match the phantom")

    print(f"{'pattern':<22}{'R':>8}{'ZF NMSE':>12}{'CG NMSE':>12}{'ZF SSIM':>10}{'CG PSNR':>10}")
    for acceleration, center in ((4, 0.08), (6, 0.06)):
        for name, mask_np in (
            ("regular", equispaced_mask(size, acceleration, center, offset=0)),
            (
                "variable_density",
                variable_density_mask(size, acceleration, center, seed=acceleration),
            ),
        ):
            mask = torch.from_numpy(mask_np)
            zf = zero_filled_rss(kspace, mask)
            cg = cg_sense(kspace, sens, mask, iterations=15)
            zf_nmse = nmse(zf, target).item()
            cg_nmse = nmse(cg, target).item()
            zf_ssim = ssim(zf, target).item()
            cg_psnr = psnr(cg, target).item()
            accel = realized_acceleration(mask_np)
            label = f"R{acceleration}_{name}"
            print(
                f"{label:<22}{accel:8.2f}{zf_nmse:12.4f}{cg_nmse:12.4f}{zf_ssim:10.3f}{cg_psnr:10.2f}"
            )
            if cg_nmse >= zf_nmse:
                raise SystemExit(f"CG-SENSE did not beat zero-filling for {label}")


if __name__ == "__main__":
    main()