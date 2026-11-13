"""Save one comparison figure for each frozen model.

Uses the first held-out test volume, central slice. Does not train and does
not change the checkpoints. TV runs with gradients enabled. The network does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import torch

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mri_recon.cs import compressed_sense
from mri_recon.fastmri_h5 import list_volumes, load_slice, split_volumes
from mri_recon.metrics import nmse
from mri_recon.model import UnrolledRecon
from mri_recon.recon import cg_sense, zero_filled_rss
from train import center_crop, prepare_slice


CASES = (
    ("r4_equispaced", "checkpoints/unrolled_eq.pt", "equispaced", 4.0),
    ("r4_variable", "checkpoints/unrolled_weak.pt", "variable", 4.0),
    ("r6_equispaced", "checkpoints/unrolled_eq_r6.pt", "equispaced", 6.0),
    ("r6_variable", "checkpoints/unrolled_vd_r6.pt", "variable", 6.0),
)


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    model = UnrolledRecon().to(device)
    saved = torch.load(path, map_location=device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def show_image(axis, picture, peak: float) -> None:
    axis.imshow(picture.detach().cpu(), cmap="gray", vmin=0, vmax=peak, aspect="equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.4)
    _train, _val, test_files = split_volumes(list_volumes(Path("data")))
    volume = test_files[0]
    with h5py.File(volume, "r") as handle:
        index = int(handle["kspace"].shape[0]) // 2
    kspace, target = load_slice(volume, index)
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"volume {volume.name} slice {index}")

    for name, checkpoint, mask_kind, acceleration in CASES:
        measured, sensitivities, mask, target_t = prepare_slice(
            kspace, target, acceleration, seed=10_000, device=device, mask_kind=mask_kind
        )
        model = load_model(Path(checkpoint), device)
        zero_fill = center_crop(zero_filled_rss(measured, mask), *target_t.shape)
        tv_image = center_crop(
            compressed_sense(measured, sensitivities, mask), *target_t.shape
        ).detach()
        cg_image = center_crop(
            cg_sense(measured, sensitivities, mask, iterations=8), *target_t.shape
        )
        with torch.no_grad():
            network = center_crop(model(measured, sensitivities, mask), *target_t.shape)
        images = {
            "target": target_t,
            "zero_fill": zero_fill,
            "tv_cs": tv_image,
            "cg_sense": cg_image,
            "unrolled": network,
        }
        peak = float(target_t.max().clamp_min(1e-8))
        error_peak = float((images["zero_fill"] - target_t).abs().max().clamp_min(1e-8))
        figure, axes = plt.subplots(2, 5, figsize=(16, 9.4))
        scores = {}
        for column, (label, image) in enumerate(images.items()):
            show_image(axes[0, column], image, peak)
            axes[0, column].set_title(label, fontsize=14, pad=10)
            axes[1, column].set_xticks([])
            axes[1, column].set_yticks([])
            for spine in axes[1, column].spines.values():
                spine.set_visible(False)
            if label == "target":
                axes[1, column].text(
                    0.5,
                    0.5,
                    "error maps\nto the right",
                    ha="center",
                    va="center",
                    fontsize=13,
                    transform=axes[1, column].transAxes,
                )
                continue
            error = (image - target_t).abs()
            axes[1, column].imshow(
                error.detach().cpu(),
                cmap="magma",
                vmin=0,
                vmax=error_peak,
                aspect="equal",
            )
            scores[column] = float(nmse(image, target_t).detach())
            print(f"{name} {label} {scores[column]:.4f}")

        figure.subplots_adjust(
            left=0.03, right=0.99, top=0.90, bottom=0.24, hspace=0.18, wspace=0.05
        )
        figure.canvas.draw()
        for column, score in scores.items():
            box = axes[1, column].get_position()
            figure.text(
                (box.x0 + box.x1) / 2,
                box.y0 - 0.012,
                f"NMSE {score:.4f}",
                ha="center",
                va="top",
                fontsize=14,
                color="black",
            )

        scale = ScalarMappable(norm=Normalize(vmin=0, vmax=error_peak), cmap="magma")
        scale.set_array([])
        colorbar_axis = figure.add_axes([0.30, 0.075, 0.40, 0.02])
        colorbar = figure.colorbar(scale, cax=colorbar_axis, orientation="horizontal")
        colorbar.ax.tick_params(labelsize=10)
        figure.text(
            0.5,
            0.028,
            "Purple = small absolute error.    Yellow = large absolute error.",
            ha="center",
            va="center",
            fontsize=14,
            color="black",
        )
        figure.suptitle(f"{volume.name}  slice {index}  {name}", fontsize=15)
        destination = out_dir / f"{name}.png"
        figure.savefig(destination, dpi=140)
        plt.close(figure)
        print(f"saved {destination}")


if __name__ == "__main__":
    main()