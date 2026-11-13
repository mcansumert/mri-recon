"""Compare zero-fill, TV compressed SENSE, CG-SENSE, and the unrolled network.

The default split is validation. The test split opens only with
--confirm-final-test. A missing checkpoint skips the unrolled row.
The mask is variable-density unless --mask equispaced is set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mri_recon.cs import compressed_sense
from mri_recon.fastmri_h5 import list_volumes, load_slice, split_volumes
from mri_recon.metrics import nmse, psnr, ssim
from mri_recon.model import UnrolledRecon
from mri_recon.recon import cg_sense, zero_filled_rss
from train import center_crop, prepare_slice


def score(estimate: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    cropped = center_crop(estimate, *target.shape)
    return (
        float(nmse(cropped, target).detach()),
        float(psnr(cropped, target).detach()),
        float(ssim(cropped, target).detach()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/unrolled.pt"))
    parser.add_argument("--max-slices", type=int, default=4)
    parser.add_argument("--acceleration", type=float, default=4.0)
    parser.add_argument("--mask", choices=("variable", "equispaced"), default="variable")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.4)

    train_files, val_files, test_files = split_volumes(list_volumes(args.data_root))
    print(f"volumes train {len(train_files)} val {len(val_files)} test {len(test_files)}")
    if args.split == "test":
        if not args.confirm_final_test:
            raise SystemExit(
                "test split is closed; pass --confirm-final-test only for the final report"
            )
        volumes = test_files
        print("reading the held-out test split once")
    else:
        volumes = val_files
        print("reading the validation split")
    if not volumes:
        raise SystemExit(f"no volumes in {args.split}")

    model = None
    if args.checkpoint.exists():
        model = UnrolledRecon().to(device)
        saved = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(saved["model"])
        model.eval()

    totals = {"zero_fill": [], "tv_cs": [], "cg_sense": [], "unrolled": []}
    seen = 0
    for volume in volumes:
        import h5py

        with h5py.File(volume, "r") as handle:
            n_slices = int(handle["kspace"].shape[0])
        for index in (n_slices // 2,):
            if seen >= args.max_slices:
                break
            kspace, target = load_slice(volume, index)
            if target is None:
                continue
            measured, sensitivities, mask, target_t = prepare_slice(
                kspace,
                target,
                args.acceleration,
                seed=10_000 + seen,
                device=device,
                mask_kind=args.mask,
            )
            totals["zero_fill"].append(score(zero_filled_rss(measured, mask), target_t))
            totals["tv_cs"].append(
                score(compressed_sense(measured, sensitivities, mask), target_t)
            )
            totals["cg_sense"].append(
                score(cg_sense(measured, sensitivities, mask, iterations=8), target_t)
            )
            if model is not None:
                with torch.no_grad():
                    prediction = model(measured, sensitivities, mask)
                totals["unrolled"].append(score(prediction, target_t))
            seen += 1
        if seen >= args.max_slices:
            break

    report = args.checkpoint.with_name(args.checkpoint.stem + f"_{args.mask}_metrics.txt")
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"split {args.split} slices {seen} acceleration {args.acceleration} mask {args.mask}",
        f"{'method':<16}{'NMSE':>10}{'PSNR':>10}{'SSIM':>10}",
    ]
    for name, rows in totals.items():
        if not rows:
            continue
        mean = torch.tensor(rows).mean(dim=0)
        lines.append(
            f"{name:<16}{float(mean[0]):10.4f}{float(mean[1]):10.2f}{float(mean[2]):10.3f}"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"saved {report}")


if __name__ == "__main__":
    main()