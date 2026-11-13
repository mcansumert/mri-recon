"""Extract a .tar.xz archive into a directory."""

import argparse
import lzma
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    count = 0
    with lzma.open(args.archive, "rb") as raw, tarfile.open(fileobj=raw, mode="r|") as archive:
        for member in archive:
            archive.extract(member, args.destination)
            if member.isfile():
                count += 1
                print(f"{count} {member.name}", flush=True)

    print(f"done {count}", flush=True)


if __name__ == "__main__":
    main()
