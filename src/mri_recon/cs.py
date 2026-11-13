"""Total-variation compressed SENSE.

Minimizes 0.5 ||E x - y||^2 + lambda TV(|x|) with gradient descent.
E = M F S is the same operator used by CG-SENSE. TV is divided by the
number of pixels so lambda stays on the same scale as the data term.
"""

from __future__ import annotations

import torch

from mri_recon.recon import apply_E, apply_EH


def isotropic_tv(magnitude: torch.Tensor) -> torch.Tensor:
    dx = magnitude[:-1, 1:] - magnitude[:-1, :-1]
    dy = magnitude[1:, :-1] - magnitude[:-1, :-1]
    return torch.sum(torch.sqrt(dx**2 + dy**2 + 1e-8)) / magnitude.numel()


def compressed_sense(
    kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 1e-3,
    steps: int = 30,
    step_size: float = 0.8,
) -> torch.Tensor:
    """Magnitude image from TV-regularized multi-coil least squares."""
    measured = kspace * mask.reshape(1, mask.shape[-1], 1)
    estimate = apply_EH(measured, sensitivities, mask).detach()
    for _ in range(steps):
        data_gradient = apply_EH(
            apply_E(estimate, sensitivities, mask) - measured,
            sensitivities,
            mask,
        ).detach()
        estimate = estimate.detach().requires_grad_(True)
        penalty = isotropic_tv(estimate.abs())
        tv_gradient, = torch.autograd.grad(penalty, estimate)
        estimate = estimate - step_size * (data_gradient + lam * tv_gradient)
    return estimate.detach().abs()