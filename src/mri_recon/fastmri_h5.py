"""Read fastMRI multi-coil volumes. One file is one acquisition.

Train and test splits are by file, never by slice of the same volume.
h5py is imported only when a file is opened, so the synthetic check does not
need it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def list_volumes(root: Path) -> list[Path]:
    return sorted(path for path in Path(root).rglob("*.h5") if path.is_file())


def split_volumes(
    paths: list[Path],
    train_fraction: float = 0.70,
    val_fraction: float = 0.10,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Fixed split of a sorted file list into train, validation, and test."""
    ordered = sorted(paths)
    count = len(ordered)
    n_train = int(round(count * train_fraction))
    n_val = int(round(count * val_fraction))
    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]
    return train, val, test


def _as_complex(array: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(array):
        return array
    names = array.dtype.names or ()
    if "real" in names and "imag" in names:
        return array["real"] + 1j * array["imag"]
    raise TypeError(f"unsupported k-space dtype {array.dtype}")


def load_slice(path: Path, slice_index: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Return k-space (coils, ky, kx) and the RSS target if the file has one."""
    import h5py

    with h5py.File(path, "r") as handle:
        kspace = _as_complex(handle["kspace"][slice_index])
        target = None
        if "reconstruction_rss" in handle:
            target = np.asarray(handle["reconstruction_rss"][slice_index], dtype=np.float32)
    return kspace, target