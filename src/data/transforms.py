from __future__ import annotations

import torch


class AddGaussianNoise:
    """Add clipped Gaussian noise to an image tensor in [0, 1]."""

    def __init__(self, mean: float = 0.0, std: float = 0.3) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            tensor.shape,
            generator=generator,
            device=tensor.device,
            dtype=tensor.dtype,
        ) * self.std + self.mean
        return (tensor + noise).clamp(0.0, 1.0)
