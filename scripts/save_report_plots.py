"""Plot the frozen test table and the training loss. No GPU and no retraining."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TEST_FILES = (
    ("R4 equispaced", Path("checkpoints/test_r4_equispaced.txt")),
    ("R4 variable", Path("checkpoints/test_r4_variable.txt")),
    ("R6 equispaced", Path("checkpoints/test_r6_equispaced.txt")),
    ("R6 variable", Path("checkpoints/test_r6_variable.txt")),
)
LOG_FILES = (
    ("R4 equispaced", Path("checkpoints/unrolled_eq_log.txt")),
    ("R4 variable", Path("checkpoints/unrolled_weak_log.txt")),
    ("R6 equispaced", Path("checkpoints/unrolled_eq_r6_log.txt")),
    ("R6 variable", Path("checkpoints/unrolled_vd_r6_log.txt")),
)
METHODS = ("zero_fill", "tv_cs", "cg_sense", "unrolled")
LOSS_LINE = re.compile(r"nmse ([0-9.]+)")


def read_table(path: Path) -> dict[str, tuple[float, float, float]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in METHODS:
            rows[parts[0]] = tuple(float(value) for value in parts[1:])
    if set(rows) != set(METHODS):
        raise SystemExit(f"missing methods in {path}")
    return rows


def read_loss(path: Path) -> list[float]:
    losses = [float(match.group(1)) for match in LOSS_LINE.finditer(path.read_text(encoding="utf-8"))]
    if not losses:
        raise SystemExit(f"no loss lines in {path}")
    return losses


def moving_average(values: list[float], window: int = 15) -> list[float]:
    totals = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        totals.append(running / min(index + 1, window))
    return totals


def main() -> None:
    tables = {label: read_table(path) for label, path in TEST_FILES}
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "test_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "method", "NMSE", "PSNR", "SSIM"))
        for label, rows in tables.items():
            for method in METHODS:
                writer.writerow((label, method, *rows[method]))
    print(f"saved {csv_path}")

    labels = list(tables)
    positions = list(range(len(labels)))
    width = 0.18
    for metric, index, filename in (("NMSE", 0, "test_nmse.png"), ("SSIM", 2, "test_ssim.png")):
        figure, axis = plt.subplots(figsize=(9, 4.5))
        for offset, method in enumerate(METHODS):
            values = [tables[label][method][index] for label in labels]
            axis.bar([pos + (offset - 1.5) * width for pos in positions], values, width=width, label=method)
        axis.set_xticks(positions, labels)
        axis.set_ylabel(metric)
        axis.legend()
        figure.tight_layout()
        destination = out_dir / filename
        figure.savefig(destination, dpi=140)
        plt.close(figure)
        print(f"saved {destination}")

    figure, axis = plt.subplots(figsize=(9, 4.5))
    for label, path in LOG_FILES:
        smoothed = moving_average(read_loss(path))
        axis.plot(smoothed, label=label)
    axis.set_xlabel("training slice")
    axis.set_ylabel("NMSE, 15-slice mean")
    axis.legend()
    figure.tight_layout()
    destination = out_dir / "train_loss.png"
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    print(f"saved {destination}")


if __name__ == "__main__":
    main()