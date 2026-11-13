"""CPU checks for the unrolled network on a synthetic phantom.

Confirms the graph, the positive data-consistency weight, and the parameter
count. An untrained network is not expected to beat CG-SENSE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mri_recon.fftc import fft2c
from mri_recon.masks import variable_density_mask
from mri_recon.metrics import nmse
from mri_recon.model import UnrolledRecon
from run_synth_check import phantom, sensitivities


def main() -> None:
    torch.manual_seed(0)
    size = 96
    image = phantom(size)
    sens = sensitivities(size, n_coils=4)
    kspace = fft2c(sens * image.unsqueeze(0))
    mask = torch.from_numpy(variable_density_mask(size, 4, 0.08, seed=1))

    model = UnrolledRecon(unrolls=2, cg_iterations=2, width=16)
    output = model(kspace, sens, mask)
    if output.shape != image.shape:
        raise SystemExit(f"unexpected shape {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise SystemExit("output has a non-finite value")

    weight = torch.nn.functional.softplus(model.raw_lambda) + 1e-4
    weight_value = float(weight.detach())
    if weight_value <= 0:
        raise SystemExit("data-consistency weight is not positive")

    loss = nmse(output, image.abs())
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise SystemExit(f"missing or non-finite gradient: {name}")

    count = sum(parameter.numel() for parameter in model.parameters())
    print(f"output_shape {tuple(output.shape)}")
    print(f"lambda {weight_value:.4f}")
    print(f"untrained_nmse {float(loss.detach()):.4f}")
    print(f"parameters {count}")
    print("model_check_ok")


if __name__ == "__main__":
    main()