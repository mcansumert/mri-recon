"""Score the trained network with its learned lambda and with a tiny lambda.

The checkpoint is only read. Validation slices only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import torch

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mri_recon.fastmri_h5 import list_volumes, load_slice, split_volumes
from mri_recon.metrics import nmse
from mri_recon.model import UnrolledRecon
from train import center_crop, prepare_slice


def mean_nmse(model, volumes, device) -> float:
    scores = []
    seen = 0
    for volume in volumes:
        with h5py.File(volume, "r") as handle:
            index = int(handle["kspace"].shape[0]) // 2
        kspace, target = load_slice(volume, index)
        measured, sensitivities, mask, target_t = prepare_slice(
            kspace, target, 4.0, seed=10_000 + seen, device=device
        )
        with torch.no_grad():
            prediction = model(measured, sensitivities, mask)
        cropped = center_crop(prediction, *target_t.shape)
        scores.append(float(nmse(cropped, target_t).detach()))
        seen += 1
        if seen >= 4:
            break
    return sum(scores) / len(scores)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.4)
    _train, volumes, _test = split_volumes(list_volumes(Path("data")))
    model = UnrolledRecon().to(device)
    saved = torch.load(Path("checkpoints/unrolled.pt"), map_location=device)
    model.load_state_dict(saved["model"])
    model.eval()

    learned = float((torch.nn.functional.softplus(model.raw_lambda) + 1e-4).detach())
    print(f"learned_lambda {learned:.6f}")
    print(f"learned_nmse {mean_nmse(model, volumes, device):.4f}")

    with torch.no_grad():
        model.raw_lambda.fill_(-20.0)
    tiny = float((torch.nn.functional.softplus(model.raw_lambda) + 1e-4).detach())
    print(f"tiny_lambda {tiny:.6f}")
    print(f"tiny_nmse {mean_nmse(model, volumes, device):.4f}")


if __name__ == "__main__":
    main()