"""TV compressed SENSE should beat zero-filling on the synthetic phantom."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mri_recon.cs import compressed_sense
from mri_recon.fftc import fft2c
from mri_recon.masks import variable_density_mask
from mri_recon.metrics import nmse
from mri_recon.recon import cg_sense, zero_filled_rss
from run_synth_check import phantom, sensitivities


def main() -> None:
    torch.manual_seed(0)
    size = 96
    image = phantom(size)
    sens = sensitivities(size, n_coils=4)
    kspace = fft2c(sens * image.unsqueeze(0))
    mask = torch.from_numpy(variable_density_mask(size, 4, 0.08, seed=3))
    target = image.abs()

    zf = nmse(zero_filled_rss(kspace, mask), target).item()
    cs = nmse(compressed_sense(kspace, sens, mask), target).item()
    cg = nmse(cg_sense(kspace, sens, mask, iterations=10), target).item()
    print(f"zero_fill_nmse {zf:.4f}")
    print(f"tv_cs_nmse {cs:.4f}")
    print(f"cg_sense_nmse {cg:.4f}")
    if cs >= zf:
        raise SystemExit("TV compressed SENSE did not beat zero-filling")
    print("cs_check_ok")


if __name__ == "__main__":
    main()