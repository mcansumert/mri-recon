"""Linear reconstructions for the multi-coil operator E = M F S.

S applies coil sensitivities, F is the centered Fourier transform, and
M keeps the sampled phase-encode lines. Zero-filling ignores the missing
lines. CG-SENSE solves the normal equations EᴴE x = Eᴴy.
"""

from __future__ import annotations

import torch

from mri_recon.fftc import fft2c, ifft2c


def _phase_mask(mask: torch.Tensor, kspace: torch.Tensor) -> torch.Tensor:
    view = mask.reshape(1, mask.shape[-1], 1)
    return view.to(device=kspace.device, dtype=kspace.real.dtype)


def zero_filled_rss(kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Inverse FFT of masked multi-coil k-space, combined by root-sum-of-squares.

    kspace is complex (coils, ky, kx). mask is real (ky,).
    """
    masked = kspace * _phase_mask(mask, kspace)
    image = ifft2c(masked)
    return torch.sqrt(torch.sum(image.abs() ** 2, dim=0))


def apply_E(
    image: torch.Tensor, sensitivities: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Forward operator. image is (ky, kx), sensitivities are (coils, ky, kx)."""
    coils = sensitivities * image.unsqueeze(0)
    return fft2c(coils) * _phase_mask(mask, sensitivities)


def apply_EH(
    kspace: torch.Tensor, sensitivities: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Adjoint operator, back to a single complex image."""
    masked = kspace * _phase_mask(mask, kspace)
    image = ifft2c(masked)
    return torch.sum(image * torch.conj(sensitivities), dim=0)


def _hermitian_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(left) * right))


def conjugate_gradient(
    normal_op,
    right_hand_side: torch.Tensor,
    iterations: int = 12,
    tol: float = 1e-6,
) -> torch.Tensor:
    """CG for a Hermitian positive semi-definite operator on complex images."""
    estimate = torch.zeros_like(right_hand_side)
    residual = right_hand_side - normal_op(estimate)
    direction = residual.clone()
    residual_energy = _hermitian_dot(residual, residual)
    target = tol * torch.linalg.vector_norm(right_hand_side).clamp_min(1e-12)
    for _ in range(iterations):
        if torch.sqrt(residual_energy.clamp_min(0)) <= target:
            break
        mapped = normal_op(direction)
        denominator = _hermitian_dot(direction, mapped)
        if float(torch.abs(denominator).detach()) < 1e-20:
            break
        step = residual_energy / denominator
        estimate = estimate + step * direction
        residual = residual - step * mapped
        next_energy = _hermitian_dot(residual, residual)
        direction = residual + (next_energy / residual_energy) * direction
        residual_energy = next_energy
    return estimate


def cg_sense(
    kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    mask: torch.Tensor,
    iterations: int = 12,
) -> torch.Tensor:
    """Magnitude image from CG-SENSE with known coil sensitivities."""

    def normal_op(image: torch.Tensor) -> torch.Tensor:
        return apply_EH(apply_E(image, sensitivities, mask), sensitivities, mask)

    rhs = apply_EH(kspace, sensitivities, mask)
    estimate = conjugate_gradient(normal_op, rhs, iterations=iterations)
    return estimate.abs()