"""A few CPU training steps on one synthetic phantom.

Checks that the unrolled loss decreases. Does not read fastMRI and does not
use the shared GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

torch.set_num_threads(2)

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
    mask = torch.from_numpy(variable_density_mask(size, 4, 0.08, seed=2))
    target = image.abs()
    model = UnrolledRecon(unrolls=2, cg_iterations=2, width=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for step in range(16):
        optimizer.zero_grad(set_to_none=True)
        output = model(kspace, sens, mask)
        loss = nmse(output, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step in (0, 8, 15):
            print(f"step {step} nmse {losses[-1]:.4f}")

    if losses[-1] >= losses[0]:
        raise SystemExit("loss did not decrease")
    print("synth_train_ok")


if __name__ == "__main__":
    main()