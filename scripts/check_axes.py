import sys
from pathlib import Path

import h5py
import torch

sys.path.insert(0, "/work/src")
from mri_recon.fftc import ifft2c

path = sorted(Path("data").rglob("*.h5"))[0]
handle = h5py.File(path, "r")
print("file", path.name)
print("attrs", {key: handle.attrs[key] for key in handle.attrs})
kspace = torch.from_numpy(handle["kspace"][handle["kspace"].shape[0] // 2])
image = ifft2c(kspace).abs().sum(0)
height, width = image.shape

def bands(picture, axis, inner):
    outer = (picture.shape[axis] - inner) // 2
    if axis == 0:
        outer_energy = picture[:outer].sum() + picture[-outer:].sum()
        inner_energy = picture[outer:-outer].sum()
    else:
        outer_energy = picture[:, :outer].sum() + picture[:, -outer:].sum()
        inner_energy = picture[:, outer:-outer].sum()
    return float(outer_energy), float(inner_energy)

print("image", tuple(image.shape))
for axis, name, inner in ((0, "axis0", 320), (1, "axis1", 320)):
    outer_energy, inner_energy = bands(image, axis, inner)
    print(f"{name} length {image.shape[axis]} outer {outer_energy:.3e} inner {inner_energy:.3e} ratio {outer_energy / inner_energy:.4f}")