"""Check the calibration estimate against known synthetic sensitivities."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mri_recon.coils import estimate_sensitivities
from mri_recon.fftc import fft2c
from run_synth_check import phantom, sensitivities


def main() -> None:
    torch.manual_seed(0)
    size = 96
    image = phantom(size)
    true_sens = sensitivities(size, n_coils=4)
    kspace = fft2c(true_sens * image.unsqueeze(0))
    estimated = estimate_sensitivities(kspace, calibration_lines=24)

    support = image.abs() > 0.2
    true_mag = true_sens.abs()[:, support]
    estimated_mag = estimated.abs()[:, support]
    true_centered = true_mag - true_mag.mean()
    estimated_centered = estimated_mag - estimated_mag.mean()
    correlation = torch.sum(true_centered * estimated_centered) / (
        torch.linalg.vector_norm(true_centered) * torch.linalg.vector_norm(estimated_centered)
    )
    value = float(correlation.detach())
    print(f"magnitude_correlation {value:.4f}")
    if value < 0.9:
        raise SystemExit("sensitivity estimate does not match the phantom")
    print("coil_check_ok")


if __name__ == "__main__":
    main()