"""Centered orthogonal 2D FFT used as the MRI Fourier operator."""

from __future__ import annotations

import torch


def fft2c(image: torch.Tensor) -> torch.Tensor:
    """Image-domain tensor to k-space. Transform is over the last two axes."""
    shifted = torch.fft.ifftshift(image, dim=(-2, -1))
    spectrum = torch.fft.fft2(shifted, norm="ortho")
    return torch.fft.fftshift(spectrum, dim=(-2, -1))


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    """K-space tensor to image domain. Transform is over the last two axes."""
    shifted = torch.fft.ifftshift(kspace, dim=(-2, -1))
    image = torch.fft.ifft2(shifted, norm="ortho")
    return torch.fft.fftshift(image, dim=(-2, -1))