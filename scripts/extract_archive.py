import lzma
import tarfile
from pathlib import Path

src = Path(r"E:\M.CansuMert\mri-recon\data\knee_multicoil_val.tar.xz")
dst = Path(r"E:\M.CansuMert\mri-recon\data\knee_multicoil_val")
dst.mkdir(parents=True, exist_ok=True)

count = 0
with lzma.open(src, "rb") as raw, tarfile.open(fileobj=raw, mode="r|") as archive:
    for member in archive:
        archive.extract(member, dst)
        if member.isfile():
            count += 1
            print(f"{count} {member.name}", flush=True)

print(f"done {count}", flush=True)