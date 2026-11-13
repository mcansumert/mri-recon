"""Coil sensitivities from the fully sampled center of k-space.

fastMRI stores multi-coil k-space, not the sensitivity matrix S.
The central phase-encode lines are kept and inverse-transformed.
Each coil image is then divided by the root-sum-of-squares image.
"""

from __future__ import annotations

import torch

from mri_recon.fftc import ifft2c


def estimate_sensitivities(
    kspace: torch.Tensor,
    calibration_lines: int = 24,
) -> torch.Tensor:
    """Return S with shape (coils, ky, kx) and unit RSS at each pixel.

    kspace is complex (coils, ky, kx). calibration_lines is the width of
    the central block used for the estimate.
    """
    if calibration_lines < 1:
        raise ValueError("calibration_lines must be positive")
    num_lines = kspace.shape[-2]
    if calibration_lines > num_lines:
        calibration_lines = num_lines
    center = num_lines // 2
    half = calibration_lines // 2
    window = torch.zeros(num_lines, dtype=kspace.real.dtype, device=kspace.device)
    start = max(0, center - half)
    window[start : start + calibration_lines] = 1
    images = ifft2c(kspace * window.view(1, -1, 1))
    rss = torch.sqrt(torch.sum(images.abs() ** 2, dim=0)).clamp_min(1e-8)
    return images / rss