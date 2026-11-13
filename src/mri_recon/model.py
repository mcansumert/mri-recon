"""MoDL-style unrolled reconstruction.

Each block reuses one CNN prior, then runs a few conjugate-gradient steps on

    (EᴴE + λI) x = Eᴴy + λ z

E = M F S is the same operator as in the classical SENSE solve. Coil
sensitivities are an input. The last convolution starts at zero and lambda
starts small, so an untrained network stays near the SENSE solution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mri_recon.recon import apply_E, apply_EH, conjugate_gradient


def complex_to_channels(image: torch.Tensor) -> torch.Tensor:
    return torch.stack((image.real, image.imag), dim=0)


def channels_to_complex(channels: torch.Tensor) -> torch.Tensor:
    return torch.complex(channels[0], channels[1])


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        updated = self.conv2(F.relu(self.conv1(features)))
        return features + updated


class Denoiser(nn.Module):
    """Shared prior. Input and output are a single complex image."""

    def __init__(self, width: int = 32, blocks: int = 5) -> None:
        super().__init__()
        self.head = nn.Conv2d(2, width, kernel_size=3, padding=1)
        self.body = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.tail = nn.Conv2d(width, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = complex_to_channels(image).unsqueeze(0)
        features = self.body(F.relu(self.head(features)))
        return channels_to_complex(self.tail(features).squeeze(0))


class UnrolledRecon(nn.Module):
    def __init__(self, unrolls: int = 5, cg_iterations: int = 4, width: int = 32) -> None:
        super().__init__()
        self.unrolls = unrolls
        self.cg_iterations = cg_iterations
        self.denoiser = Denoiser(width=width)
        self.raw_lambda = nn.Parameter(torch.tensor(-4.0))

    def data_consistency(
        self,
        prior: torch.Tensor,
        kspace: torch.Tensor,
        sensitivities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        lam = F.softplus(self.raw_lambda) + 1e-4

        def normal_op(image: torch.Tensor) -> torch.Tensor:
            measured = apply_EH(apply_E(image, sensitivities, mask), sensitivities, mask)
            return measured + lam * image

        right_hand_side = apply_EH(kspace, sensitivities, mask) + lam * prior
        return conjugate_gradient(normal_op, right_hand_side, iterations=self.cg_iterations)

    def forward(
        self,
        kspace: torch.Tensor,
        sensitivities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        image = apply_EH(kspace, sensitivities, mask)
        for _ in range(self.unrolls):
            prior = self.denoiser(image)
            image = self.data_consistency(prior, kspace, sensitivities, mask)
        return image.abs()