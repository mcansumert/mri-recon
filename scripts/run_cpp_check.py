"""Compare the C++ coil kernel with PyTorch.

Do not run until sense_kernel has been compiled. A missing module is
reported and is not treated as a passed check.
"""

from __future__ import annotations

import sys

import torch


def as_float(real: torch.Tensor, imag: torch.Tensor) -> tuple[object, object]:
    return real.numpy().astype("float32"), imag.numpy().astype("float32")


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)))


def main() -> None:
    try:
        import sense_kernel
    except ImportError:
        print("sense_kernel_not_built")
        raise SystemExit(2)

    torch.manual_seed(0)
    coils, height, width = 4, 16, 12
    image = torch.randn(height, width, dtype=torch.complex64)
    sens = torch.randn(coils, height, width, dtype=torch.complex64)
    kspace = torch.randn(coils, height, width, dtype=torch.complex64)
    mask = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1], dtype=torch.float32)

    coil_r, coil_i = sense_kernel.image_to_coils(*as_float(image.real, image.imag), *as_float(sens.real, sens.imag))
    coils_cpp = torch.from_numpy(coil_r) + 1j * torch.from_numpy(coil_i)
    coils_ref = sens * image.unsqueeze(0)
    forward_error = max_abs(coils_cpp, coils_ref)

    image_r, image_i = sense_kernel.coils_to_image(*as_float(sens.real, sens.imag), *as_float(coils_ref.real, coils_ref.imag))
    image_cpp = torch.from_numpy(image_r) + 1j * torch.from_numpy(image_i)
    image_ref = torch.sum(coils_ref * torch.conj(sens), dim=0)
    adjoint_error = max_abs(image_cpp, image_ref)

    masked_r, masked_i = sense_kernel.apply_phase_mask(*as_float(kspace.real, kspace.imag), mask.numpy().astype("float32"))
    masked_cpp = torch.from_numpy(masked_r) + 1j * torch.from_numpy(masked_i)
    masked_ref = kspace * mask.view(1, -1, 1)
    mask_error = max_abs(masked_cpp, masked_ref)

    print(f"forward_max_abs {forward_error:.3e}")
    print(f"adjoint_max_abs {adjoint_error:.3e}")
    print(f"mask_max_abs {mask_error:.3e}")
    if max(forward_error, adjoint_error, mask_error) > 1e-5:
        raise SystemExit("C++ kernel does not match PyTorch")
    print("cpp_check_ok")


if __name__ == "__main__":
    main()