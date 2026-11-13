"""Train the unrolled network on the fastMRI training split only.

The test split is never opened. CUDA is capped at 40 percent of the card.
An existing checkpoint is resumed, not discarded. The mask is variable-density
unless --mask equispaced is set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import torch

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri_recon.coils import estimate_sensitivities
from mri_recon.fastmri_h5 import list_volumes, load_slice, split_volumes
from mri_recon.masks import equispaced_mask, variable_density_mask
from mri_recon.metrics import nmse
from mri_recon.model import UnrolledRecon


def center_crop(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    y0 = max(0, (image.shape[-2] - height) // 2)
    x0 = max(0, (image.shape[-1] - width) // 2)
    return image[..., y0 : y0 + height, x0 : x0 + width]


def compress_coils(kspace: torch.Tensor, n_keep: int = 8) -> torch.Tensor:
    coils = kspace.shape[0]
    if coils <= n_keep:
        return kspace
    ky, kx = kspace.shape[-2:]
    y0 = max(0, ky // 2 - 12)
    x0 = max(0, kx // 2 - 12)
    calibration = kspace[:, y0 : y0 + 24, x0 : x0 + 24]
    matrix = calibration.reshape(coils, -1)
    covariance = matrix @ matrix.conj().transpose(0, 1)
    covariance = 0.5 * (covariance + covariance.conj().transpose(0, 1))
    _values, vectors = torch.linalg.eigh(covariance)
    basis = vectors[:, -n_keep:]
    return torch.einsum("oc,cxy->oxy", basis.conj().transpose(0, 1), kspace)


def put_readout_last(kspace: torch.Tensor, target: torch.Tensor):
    """Keep the oversampled readout fully sampled.

    fastMRI uses a readout of length 640 and crops it to 320. That axis must
    not be undersampled. It is moved to the last dimension. The other axis is
    the phase-encode direction and receives the mask. The target is moved the
    same way so the orientations still match.
    """
    if kspace.shape[-2] > kspace.shape[-1]:
        kspace = kspace.transpose(-1, -2).contiguous()
        target = target.transpose(-1, -2).contiguous()
    return kspace, target


def prepare_slice(
    kspace,
    target,
    acceleration: float,
    seed: int,
    device: torch.device,
    mask_kind: str = "variable",
):
    kspace = torch.from_numpy(kspace).to(torch.complex64)
    target = torch.from_numpy(target).to(torch.float32)
    kspace, target = put_readout_last(kspace, target)
    scale = torch.max(torch.abs(kspace)).clamp_min(1e-8)
    kspace = kspace / scale
    target = target / scale
    kspace = compress_coils(kspace)
    num_lines = kspace.shape[-2]
    if mask_kind == "equispaced":
        mask_np = equispaced_mask(num_lines, acceleration, 0.08, offset=0)
    else:
        mask_np = variable_density_mask(num_lines, acceleration, 0.08, seed=seed)
    mask = torch.from_numpy(mask_np)
    view = mask.view(1, -1, 1).to(kspace.real.dtype)
    measured = kspace * view
    sensitivities = estimate_sensitivities(measured, calibration_lines=24)
    return (
        measured.to(device),
        sensitivities.to(device),
        mask.to(device),
        target.to(device),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/unrolled.pt"))
    parser.add_argument("--max-slices", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--acceleration", type=float, default=4.0)
    parser.add_argument("--mask", choices=("variable", "equispaced"), default="variable")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.4)
    print(f"device {device}")

    files = list_volumes(args.data_root)
    train_files, val_files, test_files = split_volumes(files)
    print(f"volumes train {len(train_files)} val {len(val_files)} test {len(test_files)}")
    print("training reads the train split only")
    if not train_files:
        raise SystemExit(f"no h5 volumes under {args.data_root}")

    model = UnrolledRecon().to(device)
    if args.checkpoint.exists():
        saved = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(saved["model"])
        print(f"resumed {args.checkpoint} slices {saved.get('slices')}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.checkpoint.with_name(args.checkpoint.stem + "_log.txt")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"resume max_slices {args.max_slices} epochs {args.epochs} "
            f"acceleration {args.acceleration} mask {args.mask} lr 1e-4\n"
        )

    seen = 0
    for epoch in range(args.epochs):
        for volume in train_files:
            with h5py.File(volume, "r") as handle:
                n_slices = int(handle["kspace"].shape[0])
            first = n_slices // 4
            last = max(first + 1, (3 * n_slices) // 4)
            taken = 0
            for index in range(first, last):
                if seen >= args.max_slices or taken >= 2:
                    break
                kspace, target = load_slice(volume, index)
                if target is None:
                    continue
                measured, sensitivities, mask, target_t = prepare_slice(
                    kspace,
                    target,
                    args.acceleration,
                    seed=seen,
                    device=device,
                    mask_kind=args.mask,
                )
                optimizer.zero_grad(set_to_none=True)
                output = center_crop(model(measured, sensitivities, mask), *target_t.shape)
                loss = nmse(output, target_t)
                loss.backward()
                optimizer.step()
                message = (
                    f"epoch {epoch} slice {seen} index {index}/{n_slices} "
                    f"nmse {float(loss.detach()):.4f} file {volume.name}"
                )
                print(message)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(message + "\n")
                seen += 1
                taken += 1
            if seen >= args.max_slices:
                break
        torch.save({"model": model.state_dict(), "slices": seen}, args.checkpoint)
        print(f"saved {args.checkpoint}")
        if seen >= args.max_slices:
            break


if __name__ == "__main__":
    main()