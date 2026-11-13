"""Magnitude-image metrics. Ground truth is the second argument."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nmse(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum((estimate - target) ** 2)
    denominator = torch.sum(target**2).clamp_min(1e-12)
    return numerator / denominator


def psnr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    data_range = torch.max(target).clamp_min(1e-12)
    mse = torch.mean((estimate - target) ** 2).clamp_min(1e-12)
    return 10.0 * torch.log10(data_range**2 / mse)


def ssim(
    estimate: torch.Tensor,
    target: torch.Tensor,
    window: int = 7,
) -> torch.Tensor:
    """Mean SSIM with a uniform window. data_range is the target maximum."""
    if estimate.shape != target.shape:
        raise ValueError("estimate and target shapes differ")
    data_range = torch.max(target).clamp_min(1e-12)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    estimate = estimate.unsqueeze(0).unsqueeze(0)
    target = target.unsqueeze(0).unsqueeze(0)
    padding = window // 2

    def blur(image: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(image, window, stride=1, padding=padding)

    mu_e = blur(estimate)
    mu_t = blur(target)
    sigma_e = blur(estimate * estimate) - mu_e**2
    sigma_t = blur(target * target) - mu_t**2
    sigma_et = blur(estimate * target) - mu_e * mu_t
    numerator = (2 * mu_e * mu_t + c1) * (2 * sigma_et + c2)
    denominator = (mu_e**2 + mu_t**2 + c1) * (
        sigma_e.clamp_min(0) + sigma_t.clamp_min(0) + c2
    )
    return torch.mean(numerator / denominator)