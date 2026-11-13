"""Cartesian phase-encode masks.

The last spatial axis of k-space is the readout and stays fully sampled.
Only the phase-encode axis is undersampled. Scan time scales with the
number of phase-encode lines that are kept.

acceleration is the target ratio of all phase-encode lines to kept lines,
including the fully sampled central calibration block.
"""

from __future__ import annotations

import numpy as np


def realized_acceleration(mask: np.ndarray) -> float:
    kept = int(np.count_nonzero(mask))
    if kept == 0:
        raise ValueError("mask keeps no phase-encode lines")
    return float(mask.size) / float(kept)


def equispaced_mask(
    num_lines: int,
    acceleration: float,
    center_fraction: float,
    offset: int = 0,
) -> np.ndarray:
    """Regular parallel-imaging lattice plus a fully sampled center.

    Widely spaced outer lines leave structured aliasing and a worse
    g-factor. The center block is the calibration region.
    """
    if acceleration < 1:
        raise ValueError("acceleration must be >= 1")
    num_low = int(round(num_lines * center_fraction))
    mask = np.zeros(num_lines, dtype=np.float32)
    pad = (num_lines - num_low + 1) // 2
    mask[pad : pad + num_low] = 1.0

    denominator = num_low * acceleration - num_lines
    if abs(denominator) < 1e-6:
        return mask
    adjusted = (acceleration * (num_low - num_lines)) / denominator
    if adjusted <= 0:
        return mask
    start = int(offset) % max(int(round(adjusted)), 1)
    picked = np.arange(start, num_lines - 1, adjusted)
    mask[np.rint(picked).astype(int)] = 1.0
    return mask


def variable_density_mask(
    num_lines: int,
    acceleration: float,
    center_fraction: float,
    seed: int,
    gamma: float = 6.0,
) -> np.ndarray:
    """Variable-density pseudorandom phase encodes plus a full center.

    Outer lines are drawn without replacement. Gaussian weights peak at
    the center of k-space. The kept-line count matches the acceleration.
    """
    if acceleration < 1:
        raise ValueError("acceleration must be >= 1")
    num_keep = max(1, int(round(num_lines / acceleration)))
    num_low = min(num_keep, int(round(num_lines * center_fraction)))
    mask = np.zeros(num_lines, dtype=np.float32)
    center = num_lines // 2
    half = num_low // 2
    low_start = max(0, center - half)
    mask[low_start : low_start + num_low] = 1.0

    need = num_keep - int(mask.sum())
    if need <= 0:
        return mask

    coords = np.linspace(-1.0, 1.0, num_lines)
    weights = np.exp(-0.5 * (coords * gamma) ** 2)
    candidates = np.flatnonzero(mask == 0)
    probs = weights[candidates]
    probs = probs / probs.sum()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=need, replace=False, p=probs)
    mask[chosen] = 1.0
    return mask